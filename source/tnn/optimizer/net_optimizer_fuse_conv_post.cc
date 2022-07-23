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

#include "tnn/optimizer/net_optimizer_fuse_conv_post.h"

#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"

namespace TNN_NS {

namespace optimizer {

    // P1 priority: should be fuse after bn scale fuse
    NetOptimizerRegister<NetOptimizerFuseConvPost> g_net_optimizer_fuse_conv_post(OptPriority::P1);

    std::string NetOptimizerFuseConvPost::Strategy() {
        return kNetOptimizerFuseConvPost;
    }

    bool NetOptimizerFuseConvPost::IsSupported(const NetworkConfig &net_config) {
        auto device = net_config.device_type;
        if (device == DEVICE_METAL || device == DEVICE_OPENCL || device == DEVICE_ARM || device == DEVICE_NAIVE) {
            kLayerActivationMap[LAYER_RELU]    = ActivationType_ReLU;
            kLayerActivationMap[LAYER_RELU6]   = ActivationType_ReLU6;
            kLayerActivationMap[LAYER_SIGMOID] = ActivationType_SIGMOID_MUL;
            kLayerActivationMap[LAYER_SWISH]   = ActivationType_SIGMOID_MUL;
            return true;
        }
        if (device == DEVICE_RK_NPU) {
            kLayerActivationMap[LAYER_RELU] = ActivationType_ReLU;
            return true;
        }
        if (device == DEVICE_X86 && net_config.network_type != NETWORK_TYPE_OPENVINO) {
            kLayerActivationMap[LAYER_RELU]  = ActivationType_ReLU;
            kLayerActivationMap[LAYER_RELU6] = ActivationType_ReLU6;
            return true;
        }
        return false;
    }

    Status NetOptimizerFuseConvPost::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();
        if (count <= 1) {
            return TNN_OK;
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_fused;
        layers_fused.push_back(layers_orig[0]);

        for (int index = 1; index < count; index++) {
            auto layer_info_current = layers_orig[index];
            auto layer_info_prev    = layers_orig[index - 1];
            auto layer_current_type = layer_info_current->type;

            auto conv_param = dynamic_cast<ConvLayerParam *>(layer_info_prev->param.get());
            auto activation = kLayerActivationMap.find(layer_current_type);
            if (conv_param && activation != kLayerActivationMap.end()) {
                auto conv_output_name       = layer_info_prev->outputs[0];
                auto activation_type        = activation->second;
                bool conv_output_name_check = false;
                if (activation_type == ActivationType_SIGMOID_MUL && layer_current_type == LAYER_SIGMOID) {
                    auto sigmoid_output_name = layer_info_current->outputs[0];
                    if (index + 1 < count) {
                        auto layer_info_next = layers_orig[index + 1];
                        auto layer_next_type = layer_info_next->type;
                        auto next_inputs     = layer_info_next->inputs;
                        if (layer_next_type == LAYER_MUL && next_inputs.size() == 2 &&
                            next_inputs[0] == conv_output_name && next_inputs[1] == sigmoid_output_name) {
                            ++index;
                            layer_info_current     = layer_info_next;
                            conv_output_name_check = true;
                        }
                    }
                } else {
                    conv_output_name_check = true;
                }

                if (conv_output_name_check) {
                    // outputs of conv cannot be inputs of other layeres from index + 1
                    bool is_input_of_others = false;
                    for (int next = index + 1; next < count; next++) {
                        auto layer_info_next = layers_orig[next];
                        for (auto input_next : layer_info_next->inputs) {
                            if (conv_output_name == input_next) {
                                is_input_of_others = true;
                                break;
                            }
                        }
                        if (is_input_of_others) {
                            break;
                        }
                    }

                    // prevent fusing multiple activation layers into one conv layer
                    if (!is_input_of_others && conv_param->activation_type == ActivationType_None) {
                        if (conv_param->quantized)  {
                            // quantized conv fuse relu and relu6
                            if (activation_type == ActivationType_ReLU || activation_type == ActivationType_ReLU6) {
                                conv_param->activation_type = activation_type;
                                layer_info_prev->outputs = layer_info_current->outputs;
                            } else {
                                layers_fused.push_back(layer_info_current);
                            }
                        } else {
                            // float conv fuse
                            conv_param->activation_type = activation_type;
                            layer_info_prev->outputs    = layer_info_current->outputs;
                        }
                    } else {
                        layers_fused.push_back(layer_info_current);
                    }
                } else {
                    layers_fused.push_back(layer_info_current);
                }
            } else {
                layers_fused.push_back(layer_info_current);
            }
        }
        structure->layers = layers_fused;

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
