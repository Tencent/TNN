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

#include "tnn/optimizer/net_optimizer_fuse_conv_add.h"

#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_fuse_conv_post.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"

namespace TNN_NS {

namespace optimizer {

    // P1 priority: should be fuse after bn scale fuse
    NetOptimizerRegister<NetOptimizerFuseConvAdd> g_net_optimizer_fuse_conv_add(OptPriority::P1);

    std::string NetOptimizerFuseConvAdd::Strategy() {
        return kNetOptimizerFuseConvAdd;
    }

    bool NetOptimizerFuseConvAdd::IsSupported(const NetworkConfig &net_config) {
#ifdef TNN_CONVERTER_RUNTIME
        return false;
#else
        auto device = net_config.device_type;
        if (device == DEVICE_ARM || device == DEVICE_NAIVE) {
            auto conv_post_optimizer = NetOptimizerManager::GetNetOptimizerByName(kNetOptimizerFuseConvPost);
            if (conv_post_optimizer && conv_post_optimizer->IsSupported(net_config)) {
                conv_post_opt_ = conv_post_optimizer;
            } else {
                conv_post_opt_ = nullptr;
            }
            return true;
        }
        return false;
#endif
    }

    static bool IsPreviousLayerSupportFusion(std::shared_ptr<LayerInfo> layer_info) {
        auto param = dynamic_cast<ConvLayerParam *>(layer_info->param.get());
        if (param) {
            // only fuse conv 1x1 now
            if (param->group != 1 || param->kernels[0] != 1 || param->kernels[1] != 1 || param->strides[0] != 1 ||
                param->strides[1] != 1 || param->pads[0] != 0 || param->pads[1] != 0 || param->pads[2] != 0 ||
                param->pads[3] != 0) {
                return false;
            } else {
                return param->quantized;
            }
        }
        return false;
    }

    static bool IsCurrentLayerSupportFusion(std::shared_ptr<LayerInfo> layer_info) {
        return (layer_info->type == LAYER_ADD && layer_info->param->quantized);
    }

    static bool NeedConvAddFusion(std::shared_ptr<LayerInfo> prev, std::shared_ptr<LayerInfo> current) {
        return (IsPreviousLayerSupportFusion(prev) && IsCurrentLayerSupportFusion(current));
    }

    Status NetOptimizerFuseConvAdd::Optimize(NetStructure *structure, NetResource *resource) {
        auto ret = Status(TNN_OK);
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        // Only fuse quantized network now
        auto is_quantized_net = GetQuantizedInfoFromNetStructure(structure);
        if (!is_quantized_net) {
            return TNN_OK;
        }
        if (structure->layers.size() <= 1) {
            return TNN_OK;
        }
        // step1: do conv_post fusion before conv_add fusion
        if (conv_post_opt_) {
            ret = conv_post_opt_->Optimize(structure, resource);
            if (ret != TNN_OK) {
                return ret;
            }
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();
        if (count <= 1) {
            return TNN_OK;
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_fused;
        layers_fused.push_back(layers_orig[0]);

        // step2: do conv_add fusion
        for (int index = 1; index < count; index++) {
            auto layer_info_current = layers_orig[index];
            auto layer_info_prev    = layers_orig[index - 1];
            auto conv_param = dynamic_cast<ConvLayerParam *>(layer_info_prev->param.get());
            if (NeedConvAddFusion(layer_info_prev, layer_info_current)) {
                auto conv_output_name   = layer_info_prev->outputs[0];
                auto conv_inputs        = layer_info_prev->inputs;
                // inputs of add should contain conv_outputs, and others are pushed back to conv_inputs
                bool is_add_after_conv  = false;
                for (auto input_current : layer_info_current->inputs) {
                    if (conv_output_name != input_current) {
                        conv_inputs.push_back(input_current);
                    } else {
                        is_add_after_conv = true;
                    }
                }
                // outputs of conv cannot be inputs of other layeres except add
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

                if (is_add_after_conv && !is_input_of_others) {
                    layer_info_prev->outputs = layer_info_current->outputs;
                    layer_info_prev->inputs  = conv_inputs;
                    if (conv_param->activation_type == ActivationType_None) {
                        conv_param->fusion_type  = FusionType_Conv_Add_Activation;
                    } else {
                        conv_param->fusion_type  = FusionType_Conv_Activation_Add;
                    }
                } else {
                    layers_fused.push_back(layer_info_current);
                }
            } else {
                layers_fused.push_back(layer_info_current);
            }
        }
        structure->layers = layers_fused;

        // step3: do conv_post fusion after conv_add fusion
        if (conv_post_opt_) {
            ret = conv_post_opt_->Optimize(structure, resource);
        }

        return ret;
    }

}  // namespace optimizer

}  // namespace TNN_NS
