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

#include "tnn/optimizer/net_optimizer_cbam_fused_pooling.h"

#include <algorithm>
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
    NetOptimizerRegister<NetOptimizerCbamFusedPooling> g_net_optimizer_cbam_fused_pooling(OptPriority::P1);

    std::string NetOptimizerCbamFusedPooling::Strategy() {
        return kNetOptimizerCbamFusedPooling;
    }

    bool NetOptimizerCbamFusedPooling::IsSupported(const NetworkConfig &net_config) {
        auto device = net_config.device_type;
        return device == DEVICE_CUDA;
    }

    static bool NeedCbamPoolingFusion(std::shared_ptr<LayerInfo> ave, std::shared_ptr<LayerInfo> max) {
        if (max->type != LayerType::LAYER_POOLING || ave->type != LayerType::LAYER_POOLING) {
            return false;
        }

        auto max_param = dynamic_cast<PoolingLayerParam *>(max->param.get());
        auto ave_param = dynamic_cast<PoolingLayerParam *>(ave->param.get());
        if (!max_param || !ave_param) {
            return false;
        }

        if (max_param->pool_type != 0 || ave_param->pool_type != 1 || max->inputs[0] != ave->inputs[0] ||
                ave_param->kernels[0] != 0 || ave_param->kernels[1] != 0) {
            return false;
        }
        return true;
    }

    Status NetOptimizerCbamFusedPooling::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count = (const int)layers_orig.size();
        if (count <= 3) {
            return TNN_OK;
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_fused;

        std::vector<int> deleteLayers;
        for (int i = 0; i < count; i++) {
            if (std::find(deleteLayers.begin(), deleteLayers.end(), i) != deleteLayers.end()) continue;
            auto layer_info_current = layers_orig[i];
            bool fused = false;
            for (int j = i + 1; j < std::min(count, i + 5); j++) {
                if (std::find(deleteLayers.begin(), deleteLayers.end(), j) != deleteLayers.end()) continue;
                auto layer_info_max = layers_orig[j];
                if (NeedCbamPoolingFusion(layer_info_current, layer_info_max)) {
                    std::shared_ptr<LayerInfo> layer_info_fused_pooling = std::make_shared<LayerInfo>();
                    layer_info_fused_pooling->type = LayerType::LAYER_CBAM_FUSED_POOLING;
                    layer_info_fused_pooling->type_str = "CbamFusedPooling";
                    layer_info_fused_pooling->name = layer_info_max->name;
                    layer_info_fused_pooling->inputs = layer_info_max->inputs;
                    layer_info_fused_pooling->outputs.push_back(layer_info_current->outputs[0]);
                    layer_info_fused_pooling->outputs.push_back(layer_info_max->outputs[0]);
                    layer_info_fused_pooling->param = layer_info_max->param;
                    layers_fused.push_back(layer_info_fused_pooling);
                    deleteLayers.push_back(j);
                    fused = true;
                    break;
                }
            }
            if (!fused) {
                layers_fused.push_back(layer_info_current);
            }
        }
        structure->layers = layers_fused;

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS

