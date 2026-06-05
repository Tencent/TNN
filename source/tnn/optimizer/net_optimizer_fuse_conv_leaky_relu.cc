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

#include "tnn/optimizer/net_optimizer_fuse_conv_leaky_relu.h"

#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

namespace optimizer {

    // P1 priority: must run after BN/Scale fold (which is also P1) but is
    // independent of NetOptimizerFuseConvPost. Both are P1; either order is
    // safe because conv_post does not match LAYER_PRELU and this optimizer
    // does not match RELU/RELU6/SIGMOID.
    NetOptimizerRegister<NetOptimizerFuseConvLeakyRelu>
        g_net_optimizer_fuse_conv_leaky_relu(OptPriority::P1);

    std::string NetOptimizerFuseConvLeakyRelu::Strategy() {
        return kNetOptimizerFuseConvLeakyRelu;
    }

    bool NetOptimizerFuseConvLeakyRelu::IsSupported(const NetworkConfig &net_config) {
#ifdef TNN_DISABLE_CONV_LEAKY_RELU_FUSION
        // Compile-time A/B switch: when this macro is defined, the optimizer
        // is registered (so the symbol exists and ABI is identical) but is
        // never invoked. This produces a "baseline" library binary that we
        // can compare against the fused build apples-to-apples.
        return false;
#else
        // Currently only the OpenCL backend has compile-time alpha support
        // in its convolution kernels. ARM/X86/Metal can be added later in
        // the same shape. We deliberately gate this on DEVICE_OPENCL to
        // avoid silently changing other backends' behavior.
        return net_config.device_type == DEVICE_OPENCL;
#endif
    }

    // Read the (single) channel-shared slope value from a PRelu resource.
    // Supports DATA_TYPE_FLOAT and DATA_TYPE_HALF buffers. Returns false if
    // the buffer is malformed / contains zero elements.
    static bool ReadChannelSharedSlope(PReluLayerResource *prelu_res, float &out_alpha) {
        if (prelu_res == nullptr) {
            return false;
        }
        RawBuffer &slope = prelu_res->slope_handle;
        if (slope.GetDataCount() <= 0) {
            return false;
        }
        if (slope.GetDataType() == DATA_TYPE_FLOAT) {
            const float *p = slope.force_to<float *>();
            if (p == nullptr) return false;
            out_alpha = p[0];
            return true;
        }
        if (slope.GetDataType() == DATA_TYPE_HALF) {
            // Convert half buffer to float once; we only need element [0].
            auto fp32 = GetFloatFromRawBuffer(slope);
            if (fp32 == nullptr) return false;
            out_alpha = fp32.get()[0];
            return true;
        }
        // Other dtypes (int, bfp16, ...) are not expected for activation slopes.
        return false;
    }

    Status NetOptimizerFuseConvLeakyRelu::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }
        if (!resource) {
            // Without NetResource we can't read PRelu slope; skip silently.
            return TNN_OK;
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count = static_cast<int>(layers_orig.size());
        if (count <= 1) {
            return TNN_OK;
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_fused;
        layers_fused.push_back(layers_orig[0]);

        int fused_count = 0;

        for (int index = 1; index < count; index++) {
            auto layer_info_current = layers_orig[index];
            auto layer_info_prev    = layers_orig[index - 1];

            // Match: prev = Convolution, current = PReLU(channel_shared=1).
            if (layer_info_current->type != LAYER_PRELU) {
                layers_fused.push_back(layer_info_current);
                continue;
            }
            auto conv_param = dynamic_cast<ConvLayerParam *>(layer_info_prev->param.get());
            auto prelu_param = dynamic_cast<PReluLayerParam *>(layer_info_current->param.get());
            if (conv_param == nullptr || prelu_param == nullptr) {
                layers_fused.push_back(layer_info_current);
                continue;
            }
            if (prelu_param->channel_shared != 1) {
                // Per-channel PReLU: not yet supported in this codebase.
                // Push through as-is (will remain unfused).
                layers_fused.push_back(layer_info_current);
                continue;
            }
            // Don't double-fuse: if the conv already has activation/fusion
            // baked in (e.g. Conv+ReLU was fused earlier), skip.
            if (conv_param->activation_type != ActivationType_None) {
                layers_fused.push_back(layer_info_current);
                continue;
            }
            // Quantized conv path doesn't support LeakyReLU activation today.
            if (conv_param->quantized) {
                layers_fused.push_back(layer_info_current);
                continue;
            }

            // The conv output must not feed any consumer other than this
            // PReLU; otherwise removing the PReLU output name break things.
            // (Same safety check as conv_post.)
            const std::string &conv_output_name = layer_info_prev->outputs[0];
            bool is_input_of_others = false;
            for (int next = index + 1; next < count && !is_input_of_others; next++) {
                for (const auto &input_next : layers_orig[next]->inputs) {
                    if (input_next == conv_output_name) {
                        is_input_of_others = true;
                        break;
                    }
                }
            }
            if (is_input_of_others) {
                layers_fused.push_back(layer_info_current);
                continue;
            }

            // Pull the slope value from NetResource.
            const std::string &prelu_layer_name = layer_info_current->name;
            auto res_iter = resource->resource_map.find(prelu_layer_name);
            if (res_iter == resource->resource_map.end() || !res_iter->second) {
                LOGD("Conv+LeakyReLU fusion skipped: no PRelu resource for %s\n",
                     prelu_layer_name.c_str());
                layers_fused.push_back(layer_info_current);
                continue;
            }
            auto prelu_res = dynamic_cast<PReluLayerResource *>(res_iter->second.get());
            float alpha = 0.0f;
            if (!ReadChannelSharedSlope(prelu_res, alpha)) {
                LOGD("Conv+LeakyReLU fusion skipped: invalid slope buffer for %s\n",
                     prelu_layer_name.c_str());
                layers_fused.push_back(layer_info_current);
                continue;
            }

            // Fuse: write alpha + activation_type onto conv, splice outputs,
            // drop the PRelu layer + its resource.
            conv_param->activation_type   = ActivationType_LeakyReLU;
            conv_param->leaky_relu_alpha  = alpha;
            layer_info_prev->outputs      = layer_info_current->outputs;
            // Remove the now-orphaned PRelu resource so subsequent passes /
            // NetResource serialization don't trip over it.
            resource->resource_map.erase(res_iter);
            ++fused_count;

            LOGD("Conv+LeakyReLU fused: conv=%s prelu=%s alpha=%f\n",
                 layer_info_prev->name.c_str(), prelu_layer_name.c_str(), alpha);
            // Do NOT push the PRelu layer; it has been merged.
        }

        structure->layers = layers_fused;

        if (fused_count > 0) {
            LOGD("NetOptimizerFuseConvLeakyRelu: fused %d Conv+LeakyReLU pair(s)\n",
                 fused_count);
        }
        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
