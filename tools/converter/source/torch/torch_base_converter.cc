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

#include "torch_base_converter.h"

namespace TNN_CONVERTER {

TorchConverterManager* TorchConverterManager::torch_converter_manager_ = nullptr;

TorchConverterManager::~TorchConverterManager() {
    for (auto& iter : torch_converter_map_) {
        delete iter.second;
    }
    torch_converter_map_.clear();
    delete torch_converter_manager_;
}
TorchConverterManager* TorchConverterManager::get() {
    if (torch_converter_manager_ == nullptr) {
        torch_converter_manager_ = new TorchConverterManager;
    }
    return torch_converter_manager_;
}
TorchBaseConverter* TorchConverterManager::serach(const std::string op_type) {
    auto iter = torch_converter_map_.find(op_type);
    if (iter == torch_converter_map_.end()) {
        return nullptr;
    }
    return iter->second;
}
void TorchConverterManager::insert(const std::string op_type, TorchBaseConverter* torch_base_converter) {
    torch_converter_map_.insert(std::make_pair(op_type, torch_base_converter));
}

TNN_NS::Status TorchBaseConverter::SeparateActivation(tnn::NetStructure& net_structure,
                                                      tnn::ActivationType activation_type) {
    if (activation_type == TNN_NS::ActivationType_None) {
        return TNN_NS::TNN_CONVERT_OK;
    }
    auto& layers                         = net_structure.layers;
    const std::string conv_output_suffix = "_midline_output";
    const std::string activation_suffix  = "_activation";
    auto& layer                          = layers.back();
    if (activation_type == TNN_NS::ActivationType_ReLU || activation_type == TNN_NS::ActivationType_ReLU6) {
        auto activation_layer = new TNN_NS::LayerInfo;
        activation_layer->type =
            activation_type == TNN_NS::ActivationType_ReLU ? TNN_NS::LAYER_RELU : TNN_NS::LAYER_RELU6;
        activation_layer->type_str = activation_type == TNN_NS::ActivationType_ReLU ? "ReLU" : "ReLU6";
        activation_layer->name     = layer->name + activation_suffix;
        activation_layer->inputs.push_back(layer->outputs[0] + conv_output_suffix);
        activation_layer->outputs.push_back(layer->outputs[0]);

        // modify layer
        layer->outputs[0] = layer->outputs[0] + conv_output_suffix;
        // create activation layer
        // create relu param
        auto activation_param       = new TNN_NS::LayerParam;
        activation_layer->param     = std::shared_ptr<TNN_NS::LayerParam>(activation_param);
        activation_param->type      = activation_layer->type_str;
        activation_param->name      = layer->name + activation_suffix;
        activation_param->quantized = false;
        if (layer->param->quantized) {
            activation_param->type      = "Quantized" + activation_param->type;
            activation_param->name      = "Quantized" + activation_param->name;
            activation_param->quantized = true;
        }
        // insert activation layer
        layers.push_back(std::shared_ptr<TNN_NS::LayerInfo>(activation_layer));
    } else {
        LOGE("TNN Converter unsupport activation function\n");
        return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
    }
    return TNN_NS::TNN_CONVERT_OK;
}

void TorchBaseConverter::InsertBlobs(TNN_NS::NetStructure& net_structure) {
    auto& cur_layer     = net_structure.layers.back();
    const auto& inputs  = cur_layer->inputs;
    const auto& outputs = cur_layer->outputs;
    for (const auto& input_name : inputs) {
        const auto& blob_name = input_name;
        net_structure.blobs.insert(blob_name);
    }
    for (const auto& output_name : outputs) {
        const auto& blob_name = output_name;
        net_structure.blobs.insert(blob_name);
    }
}
}  // namespace TNN_CONVERTER