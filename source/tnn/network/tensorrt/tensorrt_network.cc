// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <memory>

#include "tnn/device/cuda/cuda_context.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/network/tensorrt/exclusive_file.h"
#include "tnn/network/tensorrt/tensorrt_network.h"
#include "tnn/network/tensorrt/utils.h"
#include "tnn/utils/md5.h"

namespace TNN_NS {

#define MAX_SCRATCH_MEMORY (1<<31 - 1)
#define TENSORRT_SERIALIZE_VERSION "v1.0"

NetworkImplFactoryRegister<NetworkImplFactory<TensorRTNetwork_>>
    g_network_impl_tensorrt_factory_register(NETWORK_TYPE_TENSORRT);

std::unordered_map<std::string, TensorRTPluginLayerBuilder*> TensorRTNetwork_::m_plugin_layer_name_map;

TensorRTNetwork_::TensorRTNetwork_() {
    int8_mode = false;
    test_mode = false;
    m_trt_engine = nullptr;
    m_trt_context = nullptr;
    m_context_memory = nullptr;
}

TensorRTNetwork_::~TensorRTNetwork_() {
    if (m_context_memory) {
        Status ret = dynamic_cast<TensorRTBlobManager*>(blob_manager_)->MemFree(m_context_memory);
        if (ret != TNN_OK) {
            LOGE("Error deconstruct TensorRT Network\n");
        }
    }
    if (m_trt_context) {
        m_trt_context->destroy();
    }
    if (m_trt_engine) m_trt_engine->destroy();
}

Status TensorRTNetwork_::Init(NetworkConfig &net_config, ModelConfig &model_config,
        AbstractModelInterpreter* interpreter, InputShapesMap inputs_shape) {
    cudaSetDevice(net_config.device_id);
    config_ = net_config;
    DefaultModelInterpreter *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    CHECK_PARAM_NULL(default_interpreter);

    NetStructure *net_structure = default_interpreter->GetNetStructure();
    NetResource *net_resource   = default_interpreter->GetNetResource();
    CHECK_PARAM_NULL(net_structure);
    CHECK_PARAM_NULL(net_resource);

    device_ = GetDevice(net_config.device_type);
    CHECK_PARAM_NULL(device_);

    context_ = device_->CreateContext(net_config.device_id);
    CHECK_PARAM_NULL(context_);

    Status ret = context_->LoadLibrary(net_config.library_path);
    if (ret != TNN_OK) {
        return ret;
    }

    {
        // use mutex to protect net_resource and net_structure in multi-thread
        std::unique_lock<std::mutex> lck(optimize_mtx_);
        ret = optimizer::NetOptimizerManager::Optimize(net_structure, net_resource, net_config);
        if (ret != TNN_OK) {
            return ret;
        }
    }

    blob_manager_ = new TensorRTBlobManager(device_);
    ret = blob_manager_->Init(net_config, net_structure, inputs_shape, GetNetResourceDataType(net_resource));
    if (ret != TNN_OK) {
        return ret;
    }

    this->m_max_batchsize = 1;

    BlobMap inputs;
    ret = blob_manager_->GetAllInputBlobs(inputs);
    if (ret != TNN_OK) {
        LOGE("ERROR: get input blobs failed");
        return ret;
    }

    for (auto iter : inputs) {
        if (iter.second->GetBlobDesc().dims[0] > this->m_max_batchsize) {
            this->m_max_batchsize = iter.second->GetBlobDesc().dims[0];
        }
    }

    ret = InitLayers(net_structure, net_resource);
    if (ret != TNN_OK) {
        return ret;
    }

    BlobMap outputs;
    ret = blob_manager_->GetAllOutputBlobs(outputs);
    if (ret != TNN_OK) {
        LOGE("ERROR: get output blobs failed");
        return ret;
    }

    ret = blob_manager_->AllocateBlobMemory();
    if (ret != TNN_OK) {
       return ret;
    }

    if (model_config.params[0].empty() && model_config.params[1].empty()) {
        test_mode = true;
    }

    std::string cache_file_name = GetCacheFileName(model_config.params[0], model_config.params[1], inputs, outputs,
        net_config.device_id, this->m_max_batchsize, this->int8_mode, config_.precision == PRECISION_LOW);
    ExclFile *file_lock = new ExclFile(cache_file_name);

    if (test_mode || false == file_lock->Ready()) {
        ret = InitWithoutCache(inputs, outputs, cache_file_name);
        if (ret != TNN_OK) {
            return ret;
        }
    } else {
        size_t size = 0;
        std::ifstream deploy_input(cache_file_name, std::ios::binary);
        deploy_input.seekg(0, deploy_input.end);
        size = deploy_input.tellg();
        deploy_input.seekg(0, deploy_input.beg);
        char *model_stream = new char[size + 1];
        deploy_input.read(model_stream, size);
        IRuntime* runtime = createInferRuntime(m_trt_logger);
        m_trt_engine = runtime->deserializeCudaEngine(model_stream, size);

        ret = CreateExecuteContext();
        if (ret != TNN_OK)
            return ret;

        runtime->destroy();
        delete[] model_stream;
        deploy_input.close();
    }

    delete file_lock;

    int bind_num = m_trt_engine->getNbBindings();
    this->m_trt_bindings = new void*[bind_num];

    for (auto iter : inputs) {
        int index = m_trt_engine->getBindingIndex(iter.first.c_str());
        this->m_trt_bindings[index] = iter.second->GetHandle().base;
        auto dims = iter.second->GetBlobDesc().dims;
        nvinfer1::Dims4 inputDims(dims[0], dims[1], dims[2], dims[3]);
        m_trt_context->setBindingDimensions(index, inputDims);
    }

    for (auto iter : outputs) {
        int index = m_trt_engine->getBindingIndex(iter.first.c_str());
        this->m_trt_bindings[index] = iter.second->GetHandle().base;
    }

    return TNN_OK;
}

Status TensorRTNetwork_::Forward() {
    bool ret = this->m_trt_context->enqueueV2(this->m_trt_bindings,
        dynamic_cast<CudaContext*>(context_)->GetStream(), nullptr);
    if (ret != true) {
        return TNNERR_CUDA_TENSORRT_ERROR;
    }
    return context_->Synchronize();
}

Status TensorRTNetwork_::Reshape(const InputShapesMap &inputs) {
    for (auto iter : inputs) {
        Blob *blob = blob_manager_->GetBlob(iter.first);
        if (blob == nullptr) {
            LOGE("TensorRTNetwork reshape blob is empty\n");
            return Status(TNNERR_PARAM_ERR, "TensorRTNetwork reshape blob is empty");
        }
        blob->GetBlobDesc().dims = iter.second;
    }

    Status ret = TNN_OK;
    for (auto cur_layer : layers_) {
        ret = dynamic_cast<TensorRTBaseLayerBuilder*>(cur_layer)->Reshape();
        if (ret != TNN_OK) {
            return ret;
        }
    }

    for (auto iter : inputs) {
        int index = m_trt_engine->getBindingIndex(iter.first.c_str());
        auto dims = iter.second;
        nvinfer1::Dims4 inputDims(dims[0], dims[1], dims[2], dims[3]);
        m_trt_context->setBindingDimensions(index, inputDims);
    }

    BlobMap outputs;
    ret = blob_manager_->GetAllOutputBlobs(outputs);
    if (ret != TNN_OK) {
        LOGE("ERROR: get output blobs failed");
        return ret;
    }
    for (auto iter : outputs) {
        int index = m_trt_engine->getBindingIndex(iter.second->GetBlobDesc().name.c_str());
        auto trt_dims = m_trt_context->getBindingDimensions(index).d;
        auto &dims = iter.second->GetBlobDesc().dims;
        dims[0] = trt_dims[0];
        dims[1] = trt_dims[1];
        dims[2] = trt_dims[2];
        dims[3] = trt_dims[3];
    }

    return TNN_OK;
}

Status TensorRTNetwork_::ForwardAsync(Callback call_back) {
    bool ret = this->m_trt_context->enqueueV2(this->m_trt_bindings,
        dynamic_cast<CudaContext*>(context_)->GetStream(), nullptr);
    if (ret != true) {
        return TNNERR_CUDA_TENSORRT_ERROR;
    }
    return TNN_OK;
}

std::unordered_map<std::string, TensorRTPluginLayerBuilder*> TensorRTNetwork_::GetPluginLayerNameMap() {
    return m_plugin_layer_name_map;
}

Status TensorRTNetwork_::InitLayers(NetStructure *net_structure, NetResource *net_resource) {
    Status ret = TNN_OK;

    for (auto layer_info : net_structure->layers) {
        LayerType type = layer_info->type;
        TensorRTBaseLayerBuilder *cur_layer = CreateTensorRTBaseLayerBuilder(type);
        if (nullptr == cur_layer) {
            LOGE("Error: CreateLayer failed, type:%d\n", type);
            return Status(TNNERR_PARAM_ERR, "CreateLayer failed");
        }

        std::string layer_name = layer_info->name;
        cur_layer->SetLayerName(layer_name);
        // set layer nodes
        std::vector<Blob *> inputs;
        std::vector<std::string> &input_names = layer_info->inputs;
        // get input nodes
        bool is_int8_blob = layer_info->param->quantized;

        for (auto name : input_names) {
            auto blob = blob_manager_->GetBlob(name);
            if (config_.precision == PRECISION_LOW) {
                blob->GetBlobDesc().data_type = DATA_TYPE_HALF;
            }
            if (is_int8_blob) {
                auto foreign_tensor = dynamic_cast<ForeignBlob*>(blob)->GetForeignTensor();
                auto tensorrt_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor);
                if (!tensorrt_tensor->GetInt8Mode()) {
                    std::string blob_scale_name = name + "_scale_data_";
                    tensorrt_tensor->SetIntResource(
                        reinterpret_cast<IntScaleResource *>(net_resource->resource_map[blob_scale_name].get()));
                    tensorrt_tensor->SetInt8Mode(true);
                }
                this->int8_mode = true;
            }
            inputs.push_back(blob);
        }

        std::vector<Blob *> outputs;
        std::vector<std::string> &output_names = layer_info->outputs;

        for (auto name : output_names) {
            auto blob = blob_manager_->GetBlob(name);
            if (is_int8_blob) {
                auto foreign_tensor = dynamic_cast<ForeignBlob*>(blob)->GetForeignTensor();
                auto tensorrt_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor);
                if (!tensorrt_tensor->GetInt8Mode()) {
                    std::string blob_scale_name = name + "_scale_data_";
                    tensorrt_tensor->SetIntResource(
                        reinterpret_cast<IntScaleResource *>(net_resource->resource_map[blob_scale_name].get()));
                    tensorrt_tensor->SetInt8Mode(true);
                }
                this->int8_mode = true;
            }
            outputs.push_back(blob);
        }

        LayerResource *layer_resource = nullptr;
        if (net_resource->resource_map.count(layer_name) != 0 ) {
            layer_resource = net_resource->resource_map[layer_name].get();
        }

        ret = cur_layer->Init(context_, layer_info->param.get(), layer_resource, inputs, outputs, device_);
        if (ret != TNN_OK) {
            LOGE("Error Init layer %s (err: %d or 0x%X)\n", cur_layer->GetLayerName().c_str(), (int)ret, (int)ret);
            return ret;
        }

        layers_.push_back(cur_layer);
        if (cur_layer->IsPluginLayer()) {
            m_plugin_layer_name_map[layer_info->name] = dynamic_cast<TensorRTPluginLayerBuilder*>(cur_layer);
        }
        cur_layer->SetBatchSize(m_max_batchsize);
    }
    return ret;
}

Status TensorRTNetwork_::CreateExecuteContext() {
    m_trt_context = m_trt_engine->createExecutionContextWithoutDeviceMemory();
    size_t context_memory_size = m_trt_engine->getDeviceMemorySize();
    Status ret = dynamic_cast<TensorRTBlobManager*>(blob_manager_)->MemAlloc(&m_context_memory, context_memory_size);
    if (ret != TNN_OK) {
        LOGE("Error Create TensorRT execute context\n");
        return ret;
    }
    m_trt_context->setDeviceMemory(m_context_memory);
    return TNN_OK;
}

Status TensorRTNetwork_::InitWithoutCache(BlobMap &inputs, BlobMap &outputs, std::string cache_file_name) {
    auto m_trt_builder = nvinfer1::createInferBuilder(m_trt_logger);
    NetworkDefinitionCreationFlags networkFlags = 1U << static_cast<uint32_t>(
        NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    if (int8_mode) networkFlags |= 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_PRECISION);
    auto m_trt_network = m_trt_builder->createNetworkV2(networkFlags);
    auto m_trt_config = m_trt_builder->createBuilderConfig();
    auto profile = m_trt_builder->createOptimizationProfile();
    for (auto input : inputs) {
        auto foreign_blob = dynamic_cast<ForeignBlob*>(input.second);
        auto desc = input.second->GetBlobDesc();
        nvinfer1::ITensor* in_tensor = m_trt_network->addInput(desc.name.c_str(),
            nvinfer1::DataType::kFLOAT, Dims4{-1, desc.dims[1], -1, -1});
        profile->setDimensions(desc.name.c_str(), OptProfileSelector::kMIN,
            Dims4{desc.dims[0], desc.dims[1], desc.dims[2], desc.dims[3]});
        profile->setDimensions(desc.name.c_str(), OptProfileSelector::kOPT,
            Dims4{desc.dims[0], desc.dims[1], desc.dims[2], desc.dims[3]});
        profile->setDimensions(desc.name.c_str(), OptProfileSelector::kMAX,
            Dims4{desc.dims[0], desc.dims[1], desc.dims[2], desc.dims[3]});
        auto foreign_tensor = foreign_blob->GetForeignTensor();
        auto tensorrt_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor);
        if (int8_mode) {
            auto input_scale_value = tensorrt_tensor->GetIntResource()->scale_handle.force_to<float *>()[0];

            Weights input_quant_shift;
            input_quant_shift.type = nvinfer1::DataType::kFLOAT;
            input_quant_shift.values = nullptr;
            input_quant_shift.count = 0;

            Weights input_quant_scale;
            input_quant_scale.type = nvinfer1::DataType::kFLOAT;
            float* input_quant_scale_data = (float*)malloc(sizeof(float));
            *input_quant_scale_data = input_scale_value;
            input_quant_scale.values = (void*)input_quant_scale_data;
            input_quant_scale.count = 1;

            Weights input_quant_power;
            input_quant_power.type = nvinfer1::DataType::kFLOAT;
            input_quant_power.values = nullptr;
            input_quant_power.count = 0;

            auto input_quant_layer = m_trt_network->addScale(*in_tensor, ScaleMode::kUNIFORM,
                input_quant_shift, input_quant_scale, input_quant_power);
            std::string input_quant_layer_name = desc.name + "_input_quant_";
            input_quant_layer->setOutputType(0, nvinfer1::DataType::kINT8);
            input_quant_layer->setName(input_quant_layer_name.c_str());

            Weights input_dequant_shift;
            input_dequant_shift.type = nvinfer1::DataType::kFLOAT;
            input_dequant_shift.values = nullptr;
            input_dequant_shift.count = 0;

            Weights input_dequant_scale;
            input_dequant_scale.type = nvinfer1::DataType::kFLOAT;
            float* input_dequant_scale_data = (float*)malloc(sizeof(float));
            *input_dequant_scale_data = 1 / input_scale_value;
            input_dequant_scale.values = (void*)input_dequant_scale_data;
            input_dequant_scale.count = 1;

            Weights input_dequant_power;
            input_dequant_power.type = nvinfer1::DataType::kFLOAT;
            input_dequant_power.values = nullptr;
            input_dequant_power.count = 0;

            auto input_dequant_layer = m_trt_network->addScale(*(input_quant_layer->getOutput(0)),
                ScaleMode::kUNIFORM, input_dequant_shift, input_dequant_scale, input_dequant_power);
            std::string input_dequant_layer_name = desc.name + "_input_dequant_";
            input_dequant_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
            input_dequant_layer->setName(input_dequant_layer_name.c_str());
            tensorrt_tensor->SetTensor(input_dequant_layer->getOutput(0));
        } else {
            tensorrt_tensor->SetTensor(in_tensor);
        }
    }
    m_trt_config->addOptimizationProfile(profile);

    for (int layer_id = 0; layer_id < this->layers_.size(); layer_id++) {
        BaseLayer* cur_layer = this->layers_[layer_id];
        nvinfer1::ILayer *cur_trt_layer =
            dynamic_cast<TensorRTBaseLayerBuilder*>(cur_layer)->AddToNetwork(m_trt_network);
        for (int out_id = 0; out_id < cur_layer->GetOutputBlobs().size(); out_id++) {
            auto output = cur_layer->GetOutputBlobs()[out_id];
            auto foreign_blob = dynamic_cast<ForeignBlob*>(output);
            nvinfer1::ITensor* output_tensor = cur_trt_layer->getOutput(out_id);
            output_tensor->setName(output->GetBlobDesc().name.c_str());
            auto foreign_tensor = foreign_blob->GetForeignTensor();
            auto tensorrt_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor);
            tensorrt_tensor->SetTensor(output_tensor);
        }
    }

    for (auto output : outputs) {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(output.second)->GetForeignTensor();
        auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
        //Do not delete, may cause trt bug
        LOGD("shape: %d %d %d\n", tensor->getDimensions().d[0], tensor->getDimensions().d[1],
            tensor->getDimensions().d[2]);
        m_trt_network->markOutput(*tensor);
    }

    m_trt_builder->setMaxBatchSize(64);
    m_trt_config->setMaxWorkspaceSize(MAX_SCRATCH_MEMORY);
    if (config_.precision == PRECISION_LOW && !this->int8_mode) {
        m_trt_config->setFlag(BuilderFlag::kFP16);
    }
    if (this->int8_mode) {
        m_trt_config->setFlag(BuilderFlag::kINT8);
    }
    m_trt_engine = m_trt_builder->buildEngineWithConfig(*m_trt_network, *m_trt_config);
    Status ret = CreateExecuteContext();
    if (ret != TNN_OK)
        return ret;
    m_trt_builder->destroy();
    m_trt_config->destroy();
    m_trt_network->destroy();

    if (!test_mode) {
        IHostMemory *model_stream = nullptr;
        model_stream = m_trt_engine->serialize();
        std::ofstream deploy_output(cache_file_name, std::ofstream::binary);
        char *model_stream_ptr = reinterpret_cast<char*>(model_stream->data());
        deploy_output.write(model_stream_ptr, model_stream->size());
        deploy_output.close();
        delete model_stream_ptr;
    }

    return TNN_OK;
}

std::string TensorRTNetwork_::GetCacheFileName(std::string cfg, std::string model, BlobMap input_map,
        BlobMap output_map, int device_id, int batchsize, bool int8_mode, bool use_fp16) {
    std::string md5_source = md5(cfg) + md5(model);

    for (auto iter : input_map) {
        char input_hw[2000];
        sprintf(input_hw, "chw:%d%d%d", iter.second->GetBlobDesc().dims[1], iter.second->GetBlobDesc().dims[2],
            iter.second->GetBlobDesc().dims[3]);
        md5_source += input_hw;
    }
    for (auto iter : output_map) {
        md5_source += iter.first;
    }

    std::string precision;
    if (int8_mode) {
        precision = "-int8";
    } else if (use_fp16) {
        precision = "-fp16";
    } else {
        precision = "";
    }

    std::string cache_file_name = "." +  md5(md5_source) + precision
        + TENSORRT_SERIALIZE_VERSION + "-b-" + std::to_string(batchsize)
        + "-" + GetGpuType(device_id) + "-" + GetTrtVersion() + GetCudaVersion()
        + ".cache";
    return cache_file_name;
}

}  //  namespace  TNN_NS

