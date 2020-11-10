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

#include "tnn/optimizer/net_optimizer_fuse_conv_relu.h"

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
    NetOptimizerRegister<NetOptimizerFuseConvRelu> g_net_optimizer_fuse_conv_relu(OptPriority::P1);

    std::string NetOptimizerFuseConvRelu::Strategy() {
        return kNetOptimizerFuseConvRelu;
    }

    bool NetOptimizerFuseConvRelu::IsSupported(const NetworkConfig &net_config) {
        auto device = net_config.device_type;
        if (device == DEVICE_METAL || device == DEVICE_OPENCL || device == DEVICE_ARM || device == DEVICE_NAIVE) {
            kLayerActivationMap[LAYER_RELU] = ActivationType_ReLU;
            kLayerActivationMap[LAYER_RELU6] = ActivationType_ReLU6;
            return true;
        }
        if (device == DEVICE_RK_NPU) {
            kLayerActivationMap[LAYER_RELU] = ActivationType_ReLU;
            return true;
        }
        return false;
    }

    Status NetOptimizerFuseConvRelu::Optimize(NetStructure *structure, NetResource *resource) {
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
                // outputs of conv cannot be inputs of other layeres except relu
                auto conv_output_name   = layer_info_prev->outputs[0];
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

                if (!is_input_of_others) {
                    conv_param->activation_type = activation->second;
                    layer_info_prev->outputs    = layer_info_current->outputs;
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
