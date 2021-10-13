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
#include <sstream>
#include <mutex>

#include "tnn/device/cuda/cuda_context.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/network/tensorrt/tensorrt_network.h"
#include "tnn/network/tensorrt/utils.h"
#include "tnn/utils/exclusive_file.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/md5.h"
#include "tnn/device/cuda/cuda_macro.h"
#include "tnn/utils/blob_dump_utils.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

#define MAX_SCRATCH_MEMORY (1<<31 - 1)
#define TENSORRT_SERIALIZE_VERSION "v1.5"

NetworkImplFactoryRegister<NetworkImplFactory<TensorRTNetwork_>>
    g_network_impl_tensorrt_factory_register(NETWORK_TYPE_TENSORRT);

std::unordered_map<std::string, TensorRTPluginLayerBuilder*> TensorRTNetwork_::m_plugin_layer_name_map;

std::mutex TensorRTNetwork_::network_mutex;

TensorRTNetwork_::TensorRTNetwork_() {
    int8_mode = false;
    test_mode = false;
    m_trt_engine = nullptr;
    m_trt_context = nullptr;
    m_context_memory = nullptr;
    m_trt_bindings = nullptr;
    device_id_ = 0;
}

TensorRTNetwork_::~TensorRTNetwork_() {
    CUDA_CHECK(cudaSetDevice(device_id_));

    if(config_.share_memory_mode == SHARE_MEMORY_MODE_SHARE_ONE_THREAD) {
        SharedMemoryManager::ReleaseSharedMemory(init_thread_id_, device_, config_.device_id, this);
    } else {
        if (m_context_memory) {
            Status ret = dynamic_cast<TensorRTBlobManager*>(blob_manager_)->MemFree(m_context_memory);
            if (ret != TNN_OK) {
                LOGE("Error deconstruct TensorRT Network\n");
            }
        }
    }

    if (m_trt_context) {
        m_trt_context->destroy();
    }

    if (m_trt_engine) m_trt_engine->destroy();

    if(m_trt_bindings) delete[] m_trt_bindings;
}

Status TensorRTNetwork_::Init(NetworkConfig &net_config, ModelConfig &model_config,
        AbstractModelInterpreter* interpreter, InputShapesMap min_inputs_shape,
        InputShapesMap max_inputs_shape, bool enable_const_folder) {
    std::unique_lock<std::mutex> lck(network_mutex);
    device_id_ = net_config.device_id;
    CUDA_CHECK(cudaSetDevice(net_config.device_id));
    config_ = net_config;
    DefaultModelInterpreter *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    CHECK_PARAM_NULL(default_interpreter);

    auto params_md5 = default_interpreter->GetParamsMd5();
    if (params_md5.size() == 0) {
        test_mode = true;
    }

    NetStructure *net_structure = default_interpreter->GetNetStructure();
    NetResource *net_resource   = default_interpreter->GetNetResource();
    net_resource_ = net_resource;
    CHECK_PARAM_NULL(net_structure);
    CHECK_PARAM_NULL(net_resource);
    net_structure_ = net_structure;
    net_resource_ = net_resource;

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

    init_thread_id_    = std::this_thread::get_id();

    blob_manager_ = new TensorRTBlobManager(device_);
    ret = blob_manager_->Init(net_config, net_structure, max_inputs_shape, GetNetResourceDataType(net_resource));
    if (ret != TNN_OK) {
        return ret;
    }

    BlobMap inputs;
    ret = blob_manager_->GetAllInputBlobs(inputs);
    if (ret != TNN_OK) {
        LOGE("ERROR: get input blobs failed");
        return ret;
    }

    ret = InitLayers(net_structure, net_resource, enable_const_folder);
    if (ret != TNN_OK) {
        return ret;
    }

    RETURN_ON_NEQ(CheckConstBlobs(), TNN_OK);

    BlobMap outputs;
    ret = blob_manager_->GetAllOutputBlobs(outputs);
    if (ret != TNN_OK) {
        LOGE("ERROR: get output blobs failed");
        return ret;
    }

    std::string cache_file_name = GetCacheFileName(params_md5, inputs, outputs, min_inputs_shape,
        net_config.device_id, this->int8_mode, config_.precision == PRECISION_LOW,
        enable_const_folder);

    std::unique_ptr<ExclFile> file_lock(new ExclFile(cache_file_name));

    if (test_mode || false == file_lock->Ready()) {
        ret = InitWithoutCache(inputs, outputs, cache_file_name, net_resource, min_inputs_shape);
        if (ret != TNN_OK) {
            return ret;
        }
    }

    if (!test_mode) {
        size_t size = 0;
        std::ifstream deploy_input(cache_file_name, std::ios::binary);
        deploy_input.seekg(0, deploy_input.end);
        size = deploy_input.tellg();
        deploy_input.seekg(0, deploy_input.beg);
        char *model_stream = new char[size + 1];
        deploy_input.read(model_stream, size);
        IRuntime* runtime = createInferRuntime(m_trt_logger);
        m_trt_engine = runtime->deserializeCudaEngine(model_stream, size);
        delete[] model_stream;
        ret = CreateExecuteContext();
        if (ret != TNN_OK)
            return ret;

        runtime->destroy();
        deploy_input.close();
    } else {
        ret = CreateExecuteContext();
        if (ret != TNN_OK)
            return ret;
    }

    int bind_num = m_trt_engine->getNbBindings();
    this->m_trt_bindings = new void*[bind_num];

    ret = ReshapeLayers();
    if (ret != TNN_OK) {
        LOGE("tensorrt network reshape layers failed\n");
        return ret;
    }

    ret = blob_manager_->AllocateBlobMemory();
    if (ret != TNN_OK) {
       return ret;
    }

    for (auto iter : outputs) {
        int index = m_trt_engine->getBindingIndex(iter.first.c_str());
        this->m_trt_bindings[index] = iter.second->GetHandle().base;
    }

    return TNN_OK;
}

Status TensorRTNetwork_::Forward() {
    CUDA_CHECK(cudaSetDevice(device_id_));
    BlobMap inputs;
    auto ret = blob_manager_->GetAllInputBlobs(inputs);
    if (ret != TNN_OK) {
        LOGE("ERROR: get input blobs failed");
        return ret;
    }

    for (auto iter : inputs) {
        int index = m_trt_engine->getBindingIndex(iter.first.c_str());
        if (index < 0) continue;
        this->m_trt_bindings[index] = iter.second->GetHandle().base;
    }

    bool trt_ret = this->m_trt_context->enqueueV2(this->m_trt_bindings,
        dynamic_cast<CudaContext*>(context_)->GetStream(), nullptr);
    if (trt_ret != true) {
        return TNNERR_CUDA_TENSORRT_ERROR;
    }
    Status status = context_->Synchronize();
    if(status != TNN_OK) {
        return status;
    }
#if (DUMP_INPUT_BLOB || DUMP_OUTPUT_BLOB)
    status = DumpAllOutputBlob();
#endif
    return status;
}

Status TensorRTNetwork_::ReshapeLayers() {
    for (auto cur_layer : layers_) {
        auto ret = dynamic_cast<TensorRTBaseLayerBuilder*>(cur_layer)->Reshape();
        if (ret != TNN_OK) {
            return ret;
        }
    }

    BlobMap inputs;
    auto ret = blob_manager_->GetAllInputBlobs(inputs);
    if (ret != TNN_OK) {
        LOGE("ERROR: get input blobs failed");
        return ret;
    }

    for (auto iter : inputs) {
        int index = m_trt_engine->getBindingIndex(iter.first.c_str());
        if (index < 0) continue;
        auto dims = blob_manager_->GetBlob(iter.first)->GetBlobDesc().dims;
        nvinfer1::Dims inputDims = ConvertToTRTDims(dims);
        m_trt_context->setBindingDimensions(index, inputDims);
        this->m_trt_bindings[index] = iter.second->GetHandle().base;
    }

    BlobMap outputs;
    ret = blob_manager_->GetAllOutputBlobs(outputs);
    if (ret != TNN_OK) {
        LOGE("ERROR: get output blobs failed");
        return ret;
    }

    for (auto blob_name : const_input_blobs_) {
        Blob *blob = blob_manager_->GetBlob(blob_name);
        auto buf = net_resource_->constant_map[blob_name];
        int index = m_trt_engine->getBindingIndex(blob_name.c_str());
        if (index < 0) continue;
        // Data is reload from const_map to blob in CudaLayerAcc::ReloadConstantBlobs
        m_trt_bindings[index] = blob->GetHandle().base;

        bool ret;
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(blob)->GetForeignTensor();
        if (std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->IsShapeTensor()) {
            auto name = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetShapeBlobName();
            auto dims = net_resource_->blob_shapes_map[name];
            ret = m_trt_context->setInputShapeBinding(index, dims.data());
        } else {
            nvinfer1::Dims inputDims = ConvertToTRTDims(buf->GetBufferDims());
            ret = m_trt_context->setBindingDimensions(index, inputDims);
        }

        if (!ret) {
            return Status(TNNERR_PARAM_ERR, "Reshape failed\n");
        }
    }

    for (auto iter : outputs) {
        int index = m_trt_engine->getBindingIndex(iter.first.c_str());
        auto trt_dims = m_trt_context->getBindingDimensions(index).d;
        DimsVector dims;
        for (int i = 0; i < m_trt_context->getBindingDimensions(index).nbDims; i++) {
            dims.push_back(trt_dims[i]);
        }
        blob_manager_->GetBlob(iter.first)->GetBlobDesc().dims = dims;
    }

    return TNN_OK;
}

Status TensorRTNetwork_::Reshape(const InputShapesMap &inputs) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    Status ret = TNN_OK;
    bool do_reshape = false;
    for (auto iter : inputs) {
        Blob *blob = blob_manager_->GetBlob(iter.first);
        if (blob == nullptr) {
            LOGE("DefaultNetwork reshape blob is empty\n");
            return Status(TNNERR_PARAM_ERR, "DefaultNetwork reshape blob is empty");
        }
        if(!DimsVectorUtils::Equal(blob->GetBlobDesc().dims, iter.second)) {
            blob->GetBlobDesc().dims = iter.second;
            do_reshape = true;
        }
    }

    if(!do_reshape) {
        return ret;
    }

    return ReshapeLayers();
}

Status TensorRTNetwork_::ForwardAsync(Callback call_back) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    BlobMap inputs;
    auto ret = blob_manager_->GetAllInputBlobs(inputs);
    if (ret != TNN_OK) {
        LOGE("ERROR: get input blobs failed");
        return ret;
    }

    for (auto iter : inputs) {
        int index = m_trt_engine->getBindingIndex(iter.first.c_str());
        if (index < 0) continue;
        this->m_trt_bindings[index] = iter.second->GetHandle().base;
    }

    bool trt_ret = this->m_trt_context->enqueueV2(this->m_trt_bindings,
        dynamic_cast<CudaContext*>(context_)->GetStream(), nullptr);
    if (trt_ret != true) {
        return TNNERR_CUDA_TENSORRT_ERROR;
    }
    Status status = TNN_OK;
#if (DUMP_INPUT_BLOB || DUMP_OUTPUT_BLOB)
    status = context_->Synchronize();
    if(status != TNN_OK) {
        return status;
    }
    status = DumpAllOutputBlob();
#endif
    return status;
}

std::unordered_map<std::string, TensorRTPluginLayerBuilder*> TensorRTNetwork_::GetPluginLayerNameMap() {
    return m_plugin_layer_name_map;
}

Status TensorRTNetwork_::InitLayers(NetStructure *net_structure, NetResource *net_resource, bool enable_const_folder) {
    Status ret = TNN_OK;

    // mark const blobs and blob data type
    auto const_blobs = net_resource->constant_map;
    for (auto layer_info : net_structure->layers) {
        std::vector<std::string> &input_names  = layer_info->inputs;
        for (auto name : input_names) {
            auto blob = blob_manager_->GetBlob(name);
            if (const_blobs.find(name) != const_blobs.end()) {
                if (runtime_model_ == RUNTIME_MODE_NORMAL) {
                    blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
                }
                blob->GetBlobDesc().data_type = const_blobs[name]->GetDataType();
            }
        }
    }

    auto const_layers = net_resource->constant_layers;
    for (auto layer_info : net_structure->layers) {
        if (runtime_model_ == RUNTIME_MODE_NORMAL && const_layers.find(layer_info->name) != const_layers.end()) {
            continue;
        }

        LayerType type = layer_info->type;
        TensorRTBaseLayerBuilder *cur_layer = CreateTensorRTBaseLayerBuilder(type);
        if (nullptr == cur_layer) {
            LOGE("Error: CreateLayer failed, type:%d\n", type);
            return Status(TNNERR_PARAM_ERR, "CreateLayer failed");
        }

        std::string layer_name = layer_info->name;
        cur_layer->SetNetwork(this);
        cur_layer->SetLayerName(layer_name);
        // set layer nodes
        std::vector<Blob *> inputs;
        std::vector<std::string> &input_names = layer_info->inputs;
        // get input nodes
        bool is_int8_blob = layer_info->param->quantized;

        for (auto name : input_names) {
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

        cur_layer->SetRuntimeMode(runtime_model_);
        cur_layer->SetConstantResource(&net_resource->constant_map);
        ret = cur_layer->Init(context_, layer_info->param.get(), layer_resource, inputs,
            outputs, device_, enable_const_folder);
        if (ret != TNN_OK) {
            LOGE("Error Init layer %s (err: %d or 0x%X)\n", cur_layer->GetLayerName().c_str(), (int)ret, (int)ret);
            return ret;
        }

        layers_.push_back(cur_layer);
        if (cur_layer->IsPluginLayer()) {
            m_plugin_layer_name_map[layer_info->name] = dynamic_cast<TensorRTPluginLayerBuilder*>(cur_layer);
        }
    }
    return ret;
}

Status TensorRTNetwork_::CreateExecuteContext() {
    m_trt_context = m_trt_engine->createExecutionContextWithoutDeviceMemory();
    context_memory_size_ = (std::max)(m_trt_engine->getDeviceMemorySize(), size_t(1024));
    Status status = TNN_OK;
    if(config_.share_memory_mode == SHARE_MEMORY_MODE_SHARE_ONE_THREAD) { 
        SharedMemory share_memory = SharedMemoryManager::GetSharedMemory(
                        context_memory_size_, init_thread_id_, device_,
                        config_.device_id, this, status);
        m_trt_context->setDeviceMemory(share_memory.shared_memory_data);
    } else if (config_.share_memory_mode == SHARE_MEMORY_MODE_DEFAULT) {
        status = dynamic_cast<TensorRTBlobManager*>(blob_manager_)->MemAlloc(&m_context_memory, context_memory_size_);
        if (status != TNN_OK) {
            LOGE("Error Create TensorRT execute context\n");
            return status;
        }
        m_trt_context->setDeviceMemory(m_context_memory);
    }
    return TNN_OK;
}

Status TensorRTNetwork_::GetForwardMemorySize(int &memory_size) {
    memory_size = context_memory_size_;
    return TNN_OK;
}

Status TensorRTNetwork_::SetForwardMemory(void *memory) {
    if (config_.share_memory_mode != SHARE_MEMORY_MODE_SET_FROM_EXTERNAL) {
        LOGE("Error Only SHARE_MEMORY_MODE_SET_FROM_EXTERNAL mode can set forward memory from external\n");
        return TNNERR_SHARE_MEMORY_MODE_NOT_SUPPORT;
    }

    m_trt_context->setDeviceMemory(memory);
    return TNN_OK;
}

Status TensorRTNetwork_::InitWithoutCache(BlobMap &inputs, BlobMap &outputs, std::string cache_file_name,
        NetResource *net_resource, const InputShapesMap &min_inputs_shape) {
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
        auto max_dims = ConvertToTRTDims(desc.dims);
        auto min_dims = max_dims;
        if (min_inputs_shape.count(desc.name) != 0) {
            min_dims = ConvertToTRTDims(min_inputs_shape.at(desc.name));
        }
        auto opt_dims = max_dims;
        auto nv_dims = ConvertToTRTDynamicDims(max_dims, min_dims);
        nvinfer1::ITensor* in_tensor = m_trt_network->addInput(desc.name.c_str(),
            ConvertToTRTDataType(desc.data_type), nv_dims);
        profile->setDimensions(desc.name.c_str(), OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(desc.name.c_str(), OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(desc.name.c_str(), OptProfileSelector::kMAX, max_dims);
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

    // Add Const_resources as inputs to tensorrt network
    for (auto blob_name : const_input_blobs_) {
        Blob *blob = blob_manager_->GetBlob(blob_name);
        auto buf = net_resource->constant_map[blob_name];
        auto foreign_blob = dynamic_cast<ForeignBlob*>(blob);
        auto foreign_tensor = foreign_blob->GetForeignTensor();

        ITensor * const_tensor = nullptr;
        DimsVector max_dims, min_dims;
        if (std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->IsShapeTensor()){
            auto shape_dims = ConvertToTRTDims(buf->GetBufferDims());
            const_tensor = m_trt_network->addInput(blob_name.c_str(),
                                            ConvertToTRTDataType(buf->GetDataType()), shape_dims);
            auto dims_blob_name = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetShapeBlobName();
            max_dims = net_resource->blob_shapes_map[dims_blob_name];
            min_dims = net_resource->min_blob_shapes_map[dims_blob_name];
            profile->setShapeValues(blob_name.c_str(), OptProfileSelector::kMIN, min_dims.data(), min_dims.size());
            profile->setShapeValues(blob_name.c_str(), OptProfileSelector::kMAX, max_dims.data(), max_dims.size());
            profile->setShapeValues(blob_name.c_str(), OptProfileSelector::kOPT, max_dims.data(), max_dims.size());
        } else {
            max_dims = net_resource->blob_shapes_map[blob_name];
            min_dims = net_resource->min_blob_shapes_map[blob_name];
            auto nv_max_dims = ConvertToTRTDims(max_dims);
            auto nv_min_dims = ConvertToTRTDims(min_dims);
            auto nv_input_dims = ConvertToTRTDynamicDims(nv_max_dims, nv_min_dims);
            const_tensor = m_trt_network->addInput(blob_name.c_str(),
                                            ConvertToTRTDataType(buf->GetDataType()), nv_input_dims);
            profile->setDimensions(blob_name.c_str(), OptProfileSelector::kMIN, nv_min_dims);
            profile->setDimensions(blob_name.c_str(), OptProfileSelector::kOPT, nv_max_dims);
            profile->setDimensions(blob_name.c_str(), OptProfileSelector::kMAX, nv_max_dims);
        }

        auto tensorrt_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor);
        tensorrt_tensor->SetTensor(const_tensor);

        {
            std::stringstream ss;
            if (std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->IsShapeTensor()){
                 ss << "shape tensor ";
            }
            ss << "<" << blob->GetBlobDesc().name << "> max_shape:[";
            for(int i: max_dims) {ss <<  i << ","; } ss << "] min_shape: [";
            for(int i: min_dims) {ss <<  i << ","; } ss << "]";
            LOGD("Add %s as input from constant_map to trt network\n", ss.str().c_str());
        }
    }
    m_trt_config->addOptimizationProfile(profile);

    // Add Const_resources as weights to tensorrt network
    for (auto blob_name : const_weight_blobs_) {
        Blob *blob = blob_manager_->GetBlob(blob_name);
        auto foreign_blob = dynamic_cast<ForeignBlob*>(blob);
        auto foreign_tensor = foreign_blob->GetForeignTensor();
        auto tensorrt_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor);
        auto buf = net_resource_->constant_map[blob_name];

        {
            std::stringstream ss;
            ss << "<" << blob->GetBlobDesc().name << "> count:" << buf->GetDataCount();
            ss << " DataType:" << buf->GetDataType() << " shape:[";
            for(int i: blob->GetBlobDesc().dims) {ss <<  i << ","; }
            ss << "]";
            LOGD("Adding %s as weights from constant_map to trt network\n", ss.str().c_str());
        }            
        
        auto const_layer = ConvertWeightToConstLayer(m_trt_network, buf.get());
        if (const_layer != nullptr) {
            const_layer->setName(blob_name.c_str());
            tensorrt_tensor->SetTensor(const_layer->getOutput(0));
        } else {
            LOGE("Add Const [%s] as weights to trt network failed\n", blob_name.c_str());
            return TNNERR_LAYER_ERR;
        }

    }

    for (int layer_id = 0; layer_id < this->layers_.size(); layer_id++) {
        BaseLayer* cur_layer = this->layers_[layer_id];
        nvinfer1::ILayer *cur_trt_layer = 
            dynamic_cast<TensorRTBaseLayerBuilder*>(cur_layer)->AddToNetwork(m_trt_network);
        if (cur_trt_layer == nullptr ) {
            LOGE("build trt layer for \"%s\" failed\n", cur_layer->GetLayerName().c_str());
            return TNNERR_LAYER_ERR;
        }
        for (int out_id = 0; out_id < cur_layer->GetOutputBlobs().size(); out_id++) {
            auto output = cur_layer->GetOutputBlobs()[out_id];
            auto foreign_blob = dynamic_cast<ForeignBlob*>(output);
            nvinfer1::ITensor* output_tensor = cur_trt_layer->getOutput(out_id);
            output_tensor->setName(output->GetBlobDesc().name.c_str());
            auto foreign_tensor = foreign_blob->GetForeignTensor();
            auto tensorrt_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor);
            tensorrt_tensor->SetTensor(output_tensor);

            {
                std::stringstream ss;
                int nbDims = output_tensor->getDimensions().nbDims;
                for( int d=0;d<nbDims;d++) ss << output_tensor->getDimensions().d[d] << ","; 
                ss << " blob shape:";
                for(auto d:output->GetBlobDesc().dims) ss << d << ",";
                LOGD("build trt layer for \"%s\", tensor shape %s\n", cur_layer->GetLayerName().c_str(), ss.str().c_str());
            }
        }
    }

    for (auto output : outputs) {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(output.second)->GetForeignTensor();
        auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
        //Do not delete, may cause trt bug
        for (int i = 0; i < tensor->getDimensions().nbDims; i++) {
            LOGD("shape: %d\n", tensor->getDimensions().d[i]);
        }
        m_trt_network->markOutput(*tensor);
    }

    m_trt_config->setMaxWorkspaceSize(MAX_SCRATCH_MEMORY);
    if (config_.precision == PRECISION_LOW && !this->int8_mode) {
        m_trt_config->setFlag(BuilderFlag::kFP16);
    }
    if (this->int8_mode) {
        m_trt_config->setFlag(BuilderFlag::kINT8);
    }
    m_trt_engine = m_trt_builder->buildEngineWithConfig(*m_trt_network, *m_trt_config);
    if (!m_trt_engine) {
        LOGE("create tensorrt engine failed\n");
        return TNNERR_CUDA_TENSORRT_ERROR;
    }
//    Status ret = CreateExecuteContext();
//    if (ret != TNN_OK)
//        return ret;
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

bool TensorRTNetwork_::IsBlobUsed(Blob* blob) {
    for (int i = 0; i < layers_.size(); i++) {
        auto inputs = layers_[i]->GetInputBlobs();
        if (std::find(inputs.begin(), inputs.end(), blob) != inputs.end()) {
            return true;
        }
    }
    return false;
}

std::string TensorRTNetwork_::GetCacheFileName(std::vector<std::string> params_md5, BlobMap input_map,
        BlobMap output_map, const InputShapesMap &min_inputs_shape, int device_id, bool int8_mode,
        bool use_fp16, bool enable_const_folder) {
    std::string md5_source = "";

    for (auto iter : params_md5) {
        md5_source += iter;
    }

    for (auto iter : input_map) {
        std::stringstream ss;
        ss << "dims:";
        for (int d : iter.second->GetBlobDesc().dims) {
            ss << d << ",";
        }
        if (min_inputs_shape.count(iter.first) != 0) {
            ss << "min_dims:";
            auto min_dims = min_inputs_shape.at(iter.first);
            for (int i = 0; i < min_dims.size(); i++) {
                ss << min_dims[i] << ",";
            }
        }
        md5_source += ss.str();
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

    std::string const_folder = enable_const_folder ? "const_folder_on" : "const_folder_off";

    std::string cache_file_name = "." +  md5(md5_source) + precision
        + TENSORRT_SERIALIZE_VERSION + "-" + GetGpuType(device_id)
        + "-" + GetTrtVersion() + GetCudaVersion()
        + "-" + const_folder + ".cache";
    return cache_file_name;
}


Status TensorRTNetwork_::DumpAllOutputBlob() {
    BlobMap outputs;
    Status ret = blob_manager_->GetAllOutputBlobs(outputs);
    if (ret != TNN_OK) {
        LOGE("ERROR: get output blobs failed");
        return ret;
    }
    for(auto output : outputs) {
        ret = DumpDeviceBlob(output.second, context_, "cuda");
        if(ret != TNN_OK) {
            LOGE("DumpDeviceBlob failed error code: %d, msg: %s \n", (int)ret, ret.description().c_str());
        }
    }
    return TNN_OK;
}

Status TensorRTNetwork_::CheckConstBlobs() {
    auto shape_differ_layers = net_resource_->shape_differ_layers;
    std::set<std::string> shape_differ_blobs;

    for (auto layer_info : net_structure_->layers) {
        if (shape_differ_layers.find(layer_info->name) != shape_differ_layers.end()) {
            for (auto name : layer_info->outputs) {
                shape_differ_blobs.insert(name);
            }
        }
    }

    std::vector<std::string> const_input_blobs;
    std::vector<std::string> const_weight_blobs;

    
    for (auto iter : net_resource_->constant_map) {
        auto blob_name = iter.first;
        Blob *blob = blob_manager_->GetBlob(blob_name);
        if (false == IsBlobUsed(blob)) {
            continue;
        }

        if (shape_differ_blobs.find(blob_name) != shape_differ_blobs.end()) {
            const_input_blobs.push_back(blob_name);
        } else {
            const_weight_blobs.push_back(blob_name);
            if (iter.second->GetDataCount() == 0) {
                auto data_type = iter.second->GetDataType();
                size_t ele_size = DataTypeUtils::GetBytesSize(data_type);
                net_resource_->constant_map[iter.first] = std::make_shared<RawBuffer>(ele_size);
                net_resource_->constant_map[iter.first]->SetDataType(data_type);
                LOGD("Updating empty buffer [%s], so trt won't crash\n", blob_name.c_str());
            }
        }
    }

    const_input_blobs_  = const_input_blobs;
    const_weight_blobs_ = const_weight_blobs;

    return TNN_OK;
}

void TensorRTNetwork_::OnSharedForwardMemoryChanged(void *memory) {
    m_trt_context->setDeviceMemory(memory);    
}

}  //  namespace  TNN_NS

