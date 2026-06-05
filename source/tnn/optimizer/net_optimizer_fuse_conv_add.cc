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
#endif
#ifdef TNN_DISABLE_CONV_ADD_FUSION
        return false;
#endif
#if !defined(TNN_CONVERTER_RUNTIME) && !defined(TNN_DISABLE_CONV_ADD_FUSION)
        auto device = net_config.device_type;
        device_ = device;
        if (device == DEVICE_ARM || device == DEVICE_NAIVE || device == DEVICE_X86) {
            auto conv_post_optimizer = NetOptimizerManager::GetNetOptimizerByName(kNetOptimizerFuseConvPost);
            if (conv_post_optimizer && conv_post_optimizer->IsSupported(net_config)) {
                conv_post_opt_ = conv_post_optimizer;
            } else {
                conv_post_opt_ = nullptr;
            }
            return true;
        }
        if (device == DEVICE_OPENCL) {
            // OpenCL fuse path runs on float/half conv, conv_post fusion (activation) is
            // independent and is registered with its own priority; we do not chain it here.
            conv_post_opt_ = nullptr;
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

    // ---------------- OpenCL fusion judgement ----------------
    // OpenCL fuse path covers OpenCLConvLayer1x1Acc and OpenCLConvLayerCommonAcc only.
    // Depthwise / Winograd are excluded because their kernels are not modified to read
    // an extra add input. The judgements below mirror the IsPrefered() logic used in
    // those acc impls so that the optimizer never folds an Add into a conv that will
    // be dispatched to depthwise or winograd at runtime.
    static bool IsOpenCLConvDepthwise(const ConvLayerParam *param) {
        // OpenCLConvLayerDepthwiseAcc::IsPrefered:
        //   group != 1 && group == input_channel && group == output_channel
        // We do not have channel information at optimizer time, but if group != 1
        // and (group == input_channel == output_channel) we conservatively assume
        // depthwise. Without channel info we can only check group != 1, which means
        // any grouped conv is excluded from fusion (safe over-approximation).
        return param->group != 1;
    }

    static bool IsOpenCLConvWinogradLikely(const ConvLayerParam *param) {
        // Mirror OpenCLConvLayerWinogradAcc::IsPrefered (see source/tnn/device/
        // opencl/acc/convolution/opencl_conv_layer_winograd_acc.cc) so that we
        // only exclude conv layers that will actually be dispatched to the
        // winograd kernel at runtime. Static (non-runtime) conditions are:
        //   group == 1
        //   kernel  == 3x3
        //   stride  == 1
        //   pad     == 1
        //   dilation == 1
        //   output_channel >  32     (strict)
        //   input_channel  >= 32
        // We deliberately skip the runtime-only checks (image2d_max_height /
        // input_w/oc ratio) which require shape information not available at
        // optimizer time; missing those checks only makes us slightly more
        // conservative (treat a borderline winograd-eligible conv as winograd
        // and skip fusion), which is safe.
        if (param->group != 1) return false;
        if (param->kernels.size() < 2 || param->strides.size() < 2 ||
            param->pads.size() < 4 || param->dialations.size() < 2) {
            return false;
        }
        if (param->kernels[0] != 3 || param->kernels[1] != 3) return false;
        if (param->strides[0] != 1 || param->strides[1] != 1) return false;
        if (param->pads[0] != 1 || param->pads[1] != 1 ||
            param->pads[2] != 1 || param->pads[3] != 1) return false;
        if (param->dialations[0] != 1 || param->dialations[1] != 1) return false;
        // Channel-based gating: a 3x3-stride1 conv with oc <= 32 (or ic < 32)
        // will fall back to the Common kernel, NOT winograd, so it is safe to
        // fuse. Without this gate the optimizer would over-conservatively skip
        // common-kernel convs that happen to be 3x3 with small channel count.
        if (param->output_channel <= 32) return false;
        if (param->input_channel  <  32) return false;
        return true;
    }

    static bool IsPreviousLayerSupportFusionOpenCL(std::shared_ptr<LayerInfo> layer_info) {
        if (layer_info->type != LAYER_CONVOLUTION) {
            return false;
        }
        auto param = dynamic_cast<ConvLayerParam *>(layer_info->param.get());
        if (!param) return false;
        // OpenCL float/half conv path; quantized OpenCL conv is unsupported by this fuse path.
        if (param->quantized) return false;
        // exclude depthwise / winograd dispatch targets
        if (IsOpenCLConvDepthwise(param)) return false;
        if (IsOpenCLConvWinogradLikely(param)) return false;
        return true;
    }

    static bool IsCurrentLayerSupportFusionOpenCL(std::shared_ptr<LayerInfo> layer_info,
                                                  NetResource *resource) {
        if (layer_info->type != LAYER_ADD) return false;
        if (layer_info->param && layer_info->param->quantized) return false;
        // Add must be a 2-input elementwise add; an Add with EltwiseLayerResource
        // (i.e. one input is a constant weight) is excluded because the fused conv
        // kernel only supports an extra runtime image input.
        if (layer_info->inputs.size() != 2) return false;
        // Reject if any of the two inputs is a constant (would not be a runtime image
        // produced by another layer).
        if (resource) {
            for (const auto &name : layer_info->inputs) {
                if (resource->constant_map.find(name) != resource->constant_map.end()) {
                    return false;
                }
            }
        }
        return true;
    }

    static bool NeedConvAddFusionOpenCL(std::shared_ptr<LayerInfo> prev,
                                        std::shared_ptr<LayerInfo> current,
                                        NetResource *resource) {
        return (IsPreviousLayerSupportFusionOpenCL(prev) &&
                IsCurrentLayerSupportFusionOpenCL(current, resource));
    }

    Status NetOptimizerFuseConvAdd::Optimize(NetStructure *structure, NetResource *resource) {
        auto ret = Status(TNN_OK);
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        const bool is_opencl_path = (device_ == DEVICE_OPENCL);

        if (!is_opencl_path) {
            // Original path: only fuse quantized network now
            auto is_quantized_net = GetQuantizedInfoFromNetStructure(structure);
            if (!is_quantized_net) {
                return TNN_OK;
            }
        }
        if (structure->layers.size() <= 1) {
            return TNN_OK;
        }
        // step1: do conv_post fusion before conv_add fusion (only for non-OpenCL path)
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
            const bool need_fuse = is_opencl_path
                                       ? NeedConvAddFusionOpenCL(layer_info_prev, layer_info_current, resource)
                                       : NeedConvAddFusion(layer_info_prev, layer_info_current);
            if (need_fuse) {
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

        // step3: do conv_post fusion after conv_add fusion (only for non-OpenCL path)
        if (conv_post_opt_) {
            ret = conv_post_opt_->Optimize(structure, resource);
        }

        return ret;
    }

}  // namespace optimizer

}  // namespace TNN_NS
