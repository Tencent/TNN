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

#include "npu_network.h"

#include <tnn/device/npu/convert/npu_base_layer_convert.h>
#include <sstream>

#include "HiAiModelManagerService.h"
#include "graph/model.h"
#include "graph/op/array_defs.h"
#include "hiai_ir_build.h"
#include "tnn/core/abstract_device.h"
#include "tnn/device/npu/convert/npu_utils.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/optimizer/net_optimizer_manager.h"

namespace tnn {

NetworkImplFactoryRegister<NetworkImplFactory<NpuNetwork>> g_network_impl_npu_factory_register(NETWORK_TYPE_NPU);

NpuNetwork::NpuNetwork() {
    model_name_ = "";
    client_     = nullptr;
    input_tensor_.clear();
    output_tensor_.clear();
}

NpuNetwork::~NpuNetwork() {
    DeInit();
}

Status NpuNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                        InputShapesMap inputs_shape) {
    auto *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    net_structure_            = default_interpreter->GetNetStructure();
    model_name_               = NpuUtils::GetFileHash(model_config);

    auto instance_input_shapes_map = net_structure_->inputs_shape_map;

    std::stringstream model_suffix_stream("");
    for (auto iter : inputs_shape) {
        if (instance_input_shapes_map.count(iter.first) > 0
            && instance_input_shapes_map[iter.first] != iter.second) {
            instance_input_shapes_map[iter.first] = iter.second;
            model_suffix_stream << "_"<< iter.first << "[";
            DimsVector value = iter.second;
            for (size_t i = 0; i < value.size(); ++i) {
                if (i != 0) {
                    model_suffix_stream << "x";
                }
                model_suffix_stream << value[i];
            }
            model_suffix_stream << "]";
        }
    }
    std::string model_suffix     = model_suffix_stream.str();
    model_name_                  = model_name_ + model_suffix;

    std::string model_path       = model_config.params[2] + model_name_ + ".om";
    LOGI("the path %s\n", model_path.c_str());

    if (net_config.device_type == DEVICE_NPU && model_config.model_type == MODEL_TYPE_TNN) {
        if (NpuUtils::FileExits(model_path)) {
            LOGI("The om file already exists in %s\n", model_config.params[2].c_str());
        } else {
            // NPU IR Build
            Status ret = IRInitLayers(net_config, interpreter, instance_input_shapes_map);
            // Build Graph
            if (ret != TNN_OK)
                return ret;
            ret = BuildModel(model_path);
            if (ret != TNN_OK) {
                printf("build om fail \n");
                return ret;
            }
        }
    } else {
        LOGE("ERROR: not support device_type %d or model type %d\n", net_config.device_type, model_config.model_type);
        return Status(TNNERR_NULL_PARAM, " not support  device_type or model type");
    }
    // Start to read from om
    client_ = std::make_shared<hiai::AiModelMngerClient>();
    assert(client_ != nullptr);
    // init Ai Model Manager Client
    hiai::AIStatus ret = client_->Init(nullptr);
    assert(ret == hiai::AI_SUCCESS);
    // get ddk version
    const char *version = client_->GetVersion();
    LOGI("ddk current version: %s", version);
    auto model_builder = std::make_shared<hiai::AiModelBuilder>(client_);

    std::vector<std::shared_ptr<hiai::AiModelDescription>> model_desc;

    hiai::MemBuffer *model_mem_buffer = model_builder->InputMemBufferCreate(model_path);
    assert(model_mem_buffer != nullptr);
    
    std::shared_ptr<hiai::AiModelDescription> desc = std::make_shared<hiai::AiModelDescription>(
        model_name_ + ".om", hiai::AiModelDescription_Frequency_HIGH, hiai::HIAI_FRAMEWORK_NONE,
        hiai::HIAI_MODELTYPE_ONLINE, hiai::AiModelDescription_DeviceType_NPU);

    desc->SetModelBuffer(model_mem_buffer->GetMemBufferData(), model_mem_buffer->GetMemBufferSize());
    // only load one model
    model_desc.push_back(desc);
    // load model
    ret = client_->Load(model_desc);
    assert(ret == hiai::AI_SUCCESS);
    // check model
    bool isModelCompatibility = true;
    ret                       = client_->CheckModelCompatibility(*desc, isModelCompatibility);
    LOGI("isModelCompatibility: %s", isModelCompatibility ? "true" : "false");
    LOGI("ret value %d", ret);
    assert(ret == hiai::AI_SUCCESS);
    // destroy unused memory
    model_builder->MemBufferDestroy(model_mem_buffer);

    input_tensor_.clear();
    output_tensor_.clear();
    std::vector<hiai::TensorDimension> input_dims;
    std::vector<hiai::TensorDimension> output_dims;
    ret = client_->GetModelIOTensorDim(model_name_ + ".om", input_dims, output_dims);
    assert(ret == hiai::AI_SUCCESS);
    if (input_dims.size() == 0) {
        LOGE("Npu the model input_dims.size() == 0");
        return TNNERR_MODEL_ERR;
    }

    for (auto dim : input_dims) {
        std::shared_ptr<hiai::AiTensor> input = std::make_shared<hiai::AiTensor>();
        ret                                   = input->Init(&dim);

        assert(ret == hiai::AI_SUCCESS);
        input_tensor_.push_back(input);
    }
    assert(input_tensor_.size() != 0);
    for (auto dim : output_dims) {
        std::shared_ptr<hiai::AiTensor> output = std::make_shared<hiai::AiTensor>();
        ret                                    = output->Init(&dim);
        assert(ret == hiai::AI_SUCCESS);
        output_tensor_.push_back(output);
    }
    assert(output_tensor_.size() != 0);
    // init input buffers
    for (int i = 0; i < input_tensor_.size(); ++i) {
        hiai::TensorDimension dims = input_dims[i];
        int n                      = dims.GetNumber();
        int c                      = dims.GetChannel();
        int h                      = dims.GetHeight();
        int w                      = dims.GetWidth();
        // add blob
        char layer_name[16];
        sprintf(layer_name, "%d", i);
        BlobDesc desc;
        desc.data_format = DATA_FORMAT_NCHW;
        desc.name        = layer_name;
        desc.dims.push_back(n);
        desc.dims.push_back(c);
        desc.dims.push_back(h);
        desc.dims.push_back(w);
        BlobHandle handle;
        handle.base                = input_tensor_[i]->GetBuffer();
        input_blob_map_[desc.name] = new Blob(desc, handle);
    }
    // init output buffers
    auto it = net_structure_->outputs.begin();
    for (int i = 0; i < output_tensor_.size(); ++i) {
        hiai::TensorDimension dims = output_dims[i];
        int n                      = dims.GetNumber();
        int c                      = dims.GetChannel();
        int h                      = dims.GetHeight();
        int w                      = dims.GetWidth();
        // add blob
        std::advance(it, i);
        std::string name = *it;
        char layer_name[name.size() + 1];
        strcpy(layer_name, name.c_str());
        BlobDesc desc;
        desc.data_format = DATA_FORMAT_NCHW;
        desc.name        = layer_name;
        desc.dims.push_back(n);
        desc.dims.push_back(c);
        desc.dims.push_back(h);
        desc.dims.push_back(w);
        BlobHandle handle;
        handle.base                 = output_tensor_[i]->GetBuffer();
        output_blob_map_[desc.name] = new Blob(desc, handle);
    }
    for (auto &layer : layers_) {
        delete (layer);
    }
    layers_.clear();
    return TNN_OK;
}  // namespace tnn

Status NpuNetwork::IRInitLayers(NetworkConfig &net_config, AbstractModelInterpreter *interpreter,
                                InputShapesMap &inputs_shape) {
    Status ret                = TNN_OK;
    auto *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    NetResource *net_resource = default_interpreter->GetNetResource();

    if (net_structure_ == NULL || net_resource == NULL) {
        LOGE("ERROR: network_ is nil, network_type may not support\n");
        return Status(TNNERR_NULL_PARAM, "network_ is nil, network_type may not support");
    }

    device_ = GetDevice(net_config.device_type);
    if (device_ == NULL) {
        return TNNERR_DEVICE_NOT_SUPPORT;
    }
    context_ = device_->CreateContext(net_config.device_id);
    if (context_ == NULL) {
        return TNNERR_DEVICE_CONTEXT_CREATE;
    }

    ret = context_->LoadLibrary(net_config.library_path);
    if (ret != TNN_OK) {
        return ret;
    }

    ret = optimizer::NetOptimizerManager::Optimize(net_structure_, net_resource, net_config.device_type);
    if (ret != TNN_OK) {
        return ret;
    }

    blob_manager_ = new BlobManager(device_);
    ret           = blob_manager_->Init(net_config, net_structure_, inputs_shape, GetNetResourceDataType(net_resource));
    if (ret != TNN_OK) {
        return ret;
    }
    // Create input operators
    ret = CreateGraphInputs(inputs_shape);
    if (ret != TNN_OK) {
        return ret;
    }
    // Init layers
    ret = InitLayers(net_resource);
    if (ret != TNN_OK) {
        return ret;
    }
    // Set Graph
    SetGraphInputsAndOutputs(inputs_shape);
    return TNN_OK;
}

Status NpuNetwork::CreateGraphInputs(InputShapesMap &input_shape_map) {
    Status ret = TNN_OK;
    // init graph input
    auto iterator = input_shape_map.begin();
    for (; iterator != input_shape_map.end(); iterator++) {
        shared_ptr<ge::op::Data> input_data;
        std::string input_name           = iterator->first;
        DimsVector dims_vector           = iterator->second;
        ret                              = NpuUtils::CreateInputData(input_data, input_name, dims_vector);
        auto input_op                    = std::make_shared<OperatorInfo>(input_data, dims_vector);
        global_operator_map_[input_name] = input_op;
    }
    return ret;
}

Status NpuNetwork::SetGraphInputsAndOutputs(InputShapesMap &input_shape_map) {
    // init graph input
    std::vector<ge::Operator> input_ops;
    std::vector<ge::Operator> output_ops;
    auto iterator = input_shape_map.begin();
    for (; iterator != input_shape_map.end(); iterator++) {
        std::string input_name = iterator->first;
        input_ops.push_back(*global_operator_map_[input_name]->GetOperator());
    }
    // init graph output
    for (auto &name : net_structure_->outputs) {
        output_ops.push_back(*global_operator_map_[name]->GetOperator());
    }
    graph_.SetInputs(input_ops).SetOutputs(output_ops);
    return TNN_OK;
}

Status NpuNetwork::BuildModel(std::string &model_path) {
    Status ret = TNN_OK;
    // Set model parameters : model name and  model name + version
    ge::Model model(model_name_, model_name_ + "_v1");
    model.SetGraph(graph_);
    // Build the ir model
    domi::HiaiIrBuild ir_build;
    domi::ModelBufferData om_model_buff;
    ir_build.CreateModelBuff(model, om_model_buff);
    bool build_ret = ir_build.BuildIRModel(model, om_model_buff);
    if (!build_ret) {
        LOGE("HIAI build model failed\n");
        return TNNERR_HIAI_API_ERROR;
    }
    // Write to OM file
    ret = NpuUtils::WriteModelFile(om_model_buff, model_path);
    if (ret != TNN_OK) {
        return ret;
    }
    ir_build.ReleaseModelBuff(om_model_buff);
    return ret;
}

Status NpuNetwork::InitLayers(NetResource *net_resource) {
    Status ret = TNN_OK;
    // loop net_structure
    for (auto layer_info : net_structure_->layers) {
        LayerType type = layer_info->type;

        NpuBaseLayer *cur_layer = CreateNpuBaseLayer(type);
        if (cur_layer == nullptr) {
            LOGE("Error: CreateLayer failed, type:%d\n", type);
            return Status(TNNERR_PARAM_ERR, "CreateLayer failed");
        }
        std::string layer_name = layer_info->name;
        cur_layer->SetLayerName(layer_name);

        // set layer nodes
        std::vector<std::shared_ptr<OperatorInfo>> input_ops;

        for (std::string &name : layer_info->inputs) {
            input_ops.push_back(global_operator_map_[name]);
        }
        LayerResource *layer_resource = net_resource->resource_map[layer_name].get();
        /*
         * cur_layer->convert
         */
        ret =
            cur_layer->Init(context_, layer_info->param.get(), layer_resource, input_ops, device_, layer_info->outputs);
        if (ret != TNN_OK) {
            LOGE("Error Init layer %s (err: %d or 0x%X)\n", cur_layer->GetLayerName().c_str(), (int)ret, (int)ret);
            return ret;
        }

        for (auto &op : cur_layer->GetOutputOps()) {
            global_operator_map_[op->GetOperator()->GetName()] = op;
        }
        layers_.push_back(cur_layer);
    }
    return ret;
}

Status NpuNetwork::GetForwardMemorySize(int &memory_size) {
    memory_size = 0;
    return TNN_OK;
}

Status NpuNetwork::SetForwardMemory(void *memory) {
    return TNN_OK;
}

Status NpuNetwork::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blob_map_;
    return TNN_OK;
}

Status NpuNetwork::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blob_map_;
    return TNN_OK;
}

Status NpuNetwork::SetDeviceAffinity(const std::vector<int> &) {
    return TNN_OK;
}

Status NpuNetwork::Reshape(const InputShapesMap &inputs) {
    return TNN_OK;
}

Status NpuNetwork::DeInit() {
    client_->UnLoadModel();
    return TNN_OK;
}

Status NpuNetwork::GetCommandQueue(void **command_queue) {
    return TNN_OK;
}

Status NpuNetwork::Forward() {
    hiai::AiContext context;
    std::string key   = "model_name";
    std::string value = model_name_ + ".om";
    context.AddPara(key, value);
    int istamp;
    hiai::AIStatus ret = client_->Process(context, input_tensor_, output_tensor_, 1000, istamp);
    if (ret != hiai::AI_SUCCESS) {
        LOGE("Forward failed! The error code :%d\n", ret);
        return TNNERR_HIAI_API_ERROR;
    }
    return TNN_OK;
}

Status NpuNetwork::ForwardAsync(Callback call_back) {
    return NpuNetwork::Forward();
}

}  // namespace tnn