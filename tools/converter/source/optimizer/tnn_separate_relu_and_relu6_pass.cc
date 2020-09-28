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

#include "tnn_optimize_pass.h"
namespace TNN_CONVERTER {

DECLARE_OPTIMIZE_PASS(SeparateReluAndRelu6);

std::string TnnOptimizeSeparateReluAndRelu6Pass::PassName() {
    return "SeparateReluAndRelu6";
}

TNN_NS::Status TnnOptimizeSeparateReluAndRelu6Pass::exec(TNN_NS::NetStructure& net_structure,
                                                         TNN_NS::NetResource& net_resource) {
    auto& layers                         = net_structure.layers;
    const std::string conv_output_suffix = "_output";
    const std::string activation_suffix  = "_activation";
    for (int i = 0; i < layers.size(); i++) {
        auto& layer = layers[i];
        if (layer->type == TNN_NS::LAYER_CONVOLUTION || layer->type == TNN_NS::LAYER_CONVOLUTION) {
            auto conv_param = dynamic_cast<TNN_NS::ConvLayerParam*>(layer->param.get());
            if (conv_param->activation_type == TNN_NS::ActivationType_None) {
                continue;
            } else if (conv_param->activation_type == TNN_NS::ActivationType_ReLU ||
                       conv_param->activation_type == TNN_NS::ActivationType_ReLU6) {
                auto activation_layer  = new TNN_NS::LayerInfo;
                activation_layer->type = conv_param->activation_type == TNN_NS::ActivationType_ReLU
                                             ? TNN_NS::LAYER_RELU
                                             : TNN_NS::LAYER_RELU6;
                activation_layer->type_str =
                    conv_param->activation_type == TNN_NS::ActivationType_ReLU ? "ReLU" : "ReLU6";
                activation_layer->name = layer->name + activation_suffix;
                activation_layer->inputs.push_back(layer->outputs[0] + conv_output_suffix);
                activation_layer->outputs.push_back(layer->outputs[0]);
                // update convolution output
                layer->outputs[0] = layer->outputs[0] + conv_output_suffix;
                // create relu param
                auto activation_param       = new TNN_NS::LayerParam;
                activation_layer->param     = std::shared_ptr<TNN_NS::LayerParam>(activation_param);
                activation_param->type      = activation_layer->type;
                activation_param->name      = layer->name + activation_suffix;
                activation_param->quantized = false;
                // insert activation layer
                layers.insert(layers.begin() + i + 1, std::shared_ptr<TNN_NS::LayerInfo>(activation_layer));
            }
        }
    }
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_OPTIMIZE_PASS(SeparateReluAndRelu6);

}  // namespace TNN_CONVERTER
