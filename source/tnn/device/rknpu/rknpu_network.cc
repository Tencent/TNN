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

#include "rknpu_network.h"

#include <tnn/device/rknpu/convert/rknpu_base_layer.h>
#include <tnn/interpreter/layer_resource_generator.h>

#include <numeric>
#include <sstream>

#include "tnn/core/abstract_device.h"
#include "tnn/device/rknpu/convert/rknpu_utils.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/optimizer/net_optimizer_manager.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<RknpuNetwork>> g_network_impl_rknpu_factory_register(NETWORK_TYPE_RK_NPU);

RknpuNetwork::RknpuNetwork() {
    input_inf_.clear();
    output_inf_.clear();
}

RknpuNetwork::~RknpuNetwork() {
    DeInit();
}

Status RknpuNetwork::Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                          InputShapesMap inputs_shape) {
    rk::nn::Graph *graph = new rk::nn::Graph();
    exector_             = std::unique_ptr<rk::nn::Exection>(new rk::nn::Exection(graph));

    auto *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    net_structure_            = default_interpreter->GetNetStructure();

    auto instance_input_shapes_map = net_structure_->inputs_shape_map;

    if (net_config.device_type == DEVICE_RK_NPU && model_config.model_type == MODEL_TYPE_TNN) {
        // RKNPU IR Build
        Status build_ret = IRInitLayers(net_config, interpreter, instance_input_shapes_map);
        if (build_ret != TNN_OK) {
            return build_ret;
        }
        int ret = exector_->Build();
        if (rk::nn::RK_SUCCESS != ret)
            return TNNERR_MODEL_ERR;
    } else {
        return Status(TNNERR_NULL_PARAM, "Rknpu not support device_type or model type");
    }

    input_inf_.clear();
    output_inf_.clear();

    // init input buffers
    input_inf_.resize(graph->GetInputs().size());
    for (int i = 0; i < input_inf_.size(); i++) {
        auto type     = graph->GetInputs()[i]->GetPrecision();
        auto dims     = graph->GetInputs()[i]->GetDims();
        uint32_t size = RknpuUtils::CalcSize(type, dims);

        input_inf_[i].index        = i;
        input_inf_[i].buf          = malloc(size);
        input_inf_[i].size         = size;
        input_inf_[i].pass_through = false;
        input_inf_[i].type         = type;
        input_inf_[i].layout       = rk::nn::DataLayoutType::NCHW;

        auto it = instance_input_shapes_map.begin();
        std::advance(it, i);

        BlobDesc desc;
        desc.device_type = DEVICE_RK_NPU;
        desc.data_format = DATA_FORMAT_NCHW;
        desc.data_type   = DATA_TYPE_FLOAT;
        desc.name        = it->first;
        for (auto dim : dims) {
            desc.dims.push_back((int)dim);
        }
        BlobHandle handle;
        handle.base                = input_inf_[i].buf;
        input_blob_map_[desc.name] = new Blob(desc, handle);
    }

    // init output buffers
    output_inf_.resize(graph->GetOutputs().size());
    for (int i = 0; i < output_inf_.size(); ++i) {
        auto type     = graph->GetOutputs()[i]->GetPrecision();
        auto dims     = graph->GetOutputs()[i]->GetDims();
        uint32_t size = RknpuUtils::CalcSize(type, dims);

        output_inf_[i].index      = i;
        output_inf_[i].buf        = malloc(size);
        output_inf_[i].size       = size;
        output_inf_[i].type       = type;
        output_inf_[i].layout     = rk::nn::DataLayoutType::NCHW;
        output_inf_[i].want_float = true;

        // add blob
        auto it = net_structure_->outputs.begin();
        std::advance(it, i);
        BlobDesc desc;
        desc.device_type = DEVICE_RK_NPU;
        desc.data_format = DATA_FORMAT_NCHW;
        desc.data_type   = DATA_TYPE_FLOAT;
        desc.name        = *it;
        for (auto dim : dims) {
            desc.dims.push_back((int)dim);
        }
        BlobHandle handle;
        handle.base                 = output_inf_[i].buf;
        output_blob_map_[desc.name] = new Blob(desc, handle);
    }
    for (auto &layer : layers_) {
        delete (layer);
    }
    layers_.clear();

    return TNN_OK;
}

Status RknpuNetwork::IRInitLayers(NetworkConfig &net_config, AbstractModelInterpreter *interpreter,
                                  InputShapesMap &inputs_shape) {
    Status ret                = TNN_OK;
    auto *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    NetResource *net_resource = default_interpreter->GetNetResource();

    if (net_structure_ == NULL || net_resource == NULL) {
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

    ret = optimizer::NetOptimizerManager::Optimize(net_structure_, net_resource, net_config);
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
    ret = ConvertLayers(net_resource);
    if (ret != TNN_OK) {
        return ret;
    }
    // Set Graph
    SetGraphInputsAndOutputs(inputs_shape);
    return TNN_OK;
}

Status RknpuNetwork::CreateGraphInputs(InputShapesMap &input_shape_map) {
    Status ret = TNN_OK;

    // init graph input
    auto iterator = input_shape_map.begin();
    for (; iterator != input_shape_map.end(); iterator++) {
        std::string input_name = iterator->first;
        DimsVector dims_vector = iterator->second;

        auto rk_input = RknpuUtils::CreateRknnTensor(exector_->GetGraph(), input_name, dims_vector, NULL,
                                                     rk::nn::TensorRole::DATA, DATA_TYPE_FLOAT);

        global_operator_map_[input_name] = rk_input;
    }

    return ret;
}

Status RknpuNetwork::SetGraphInputsAndOutputs(InputShapesMap &input_shape_map) {
    // init graph input
    std::vector<std::shared_ptr<rk::nn::Tensor>> input_tensors;
    std::vector<std::shared_ptr<rk::nn::Tensor>> output_tensors;
    auto iterator = input_shape_map.begin();
    for (; iterator != input_shape_map.end(); iterator++) {
        std::string input_name = iterator->first;
        input_tensors.push_back(global_operator_map_[input_name]);
    }
    // init graph output
    for (auto &name : net_structure_->outputs) {
        auto tensor = global_operator_map_[name];
        tensor->SetRole(rk::nn::TensorRole::DATA);
        tensor->SetPrecision(rk::nn::PrecisionType::FLOAT32);
        output_tensors.push_back(tensor);
    }

    exector_->GetGraph()->SetInputsOutputs(input_tensors, output_tensors);

    return TNN_OK;
}

Status RknpuNetwork::ConvertLayers(NetResource *net_resource) {
    Status ret = TNN_OK;
    // loop net_structure
    for (auto layer_info : net_structure_->layers) {
        LayerType type            = layer_info->type;
        RknpuBaseLayer *cur_layer = CreateRknpuBaseLayer(type);
        if (cur_layer == nullptr) {
            LOGE("Error: CreateLayer failed, type:%d\n", type);
            return Status(TNNERR_PARAM_ERR, "CreateLayer failed");
        }
        std::string layer_name = layer_info->name;
        cur_layer->SetLayerName(layer_name);

        // set layer nodes
        std::vector<std::shared_ptr<rk::nn::Tensor>> input_ops;

        for (std::string &name : layer_info->inputs) {
            input_ops.push_back(global_operator_map_[name]);
        }

        LayerResource *layer_resource = net_resource->resource_map[layer_name].get();
        /*
         * cur_layer->convert
         */
        ret = cur_layer->Init(context_, layer_info->param.get(), layer_resource, input_ops, exector_->GetGraph(),
                              layer_info->outputs);
        if (ret != TNN_OK) {
            LOGE("Error Init layer %s (err: %d or 0x%X)\n", cur_layer->GetLayerName().c_str(), (int)ret, (int)ret);
            return ret;
        }

        for (auto &op : cur_layer->GetOutputOps()) {
            global_operator_map_[op->GetName()] = op;
        }
        layers_.push_back(cur_layer);
    }
    return ret;
}

Status RknpuNetwork::GetForwardMemorySize(int &memory_size) {
    memory_size = 0;
    return TNN_OK;
}

Status RknpuNetwork::SetForwardMemory(void *memory) {
    return TNN_OK;
}

Status RknpuNetwork::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blob_map_;
    return TNN_OK;
}

Status RknpuNetwork::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blob_map_;
    return TNN_OK;
}

Status RknpuNetwork::SetDeviceAffinity(const std::vector<int> &) {
    return TNN_OK;
}

Status RknpuNetwork::Reshape(const InputShapesMap &inputs) {
    return TNN_OK;
}

Status RknpuNetwork::DeInit() {
    rk::nn::Graph *graph = exector_->GetGraph();
    delete graph;

    for (auto inf : input_inf_) {
        if (inf.buf) {
            free(inf.buf);
            inf.buf = NULL;
        }
    }
    for (auto inf : output_inf_) {
        if (inf.buf) {
            free(inf.buf);
            inf.buf = NULL;
        }
    }

    for (auto &input_blob : input_blob_map_) {
        if (input_blob.second)
            delete input_blob.second;
    }
    for (auto &output_blob : output_blob_map_) {
        if (output_blob.second)
            delete output_blob.second;
    }
    if (blob_manager_)
        delete blob_manager_;

    if (context_ != nullptr) {
        delete context_;
        context_ = nullptr;
    }

    return TNN_OK;
}

Status RknpuNetwork::GetCommandQueue(void **command_queue) {
    return TNN_OK;
}

Status RknpuNetwork::Forward() {
    exector_->SetInputs(input_inf_);
    exector_->Run();
    exector_->GetOutputs(output_inf_);
    return TNN_OK;
}

Status RknpuNetwork::ForwardAsync(Callback call_back) {
    return RknpuNetwork::Forward();
}

}  // namespace TNN_NS
