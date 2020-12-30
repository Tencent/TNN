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

#include "tnn/optimizer/net_optimizer_cbam_fused_reduce.h"

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
    NetOptimizerRegister<NetOptimizerCbamFusedReduce> g_net_optimizer_cbam_fused_reduce(OptPriority::P1);

    std::string NetOptimizerCbamFusedReduce::Strategy() {
        return kNetOptimizerCbamFusedReduce;
    }

    bool NetOptimizerCbamFusedReduce::IsSupported(const NetworkConfig &net_config) {
        auto device = net_config.device_type;
        return device == DEVICE_CUDA;
    }

    static bool NeedCbamReduceFusion(std::shared_ptr<LayerInfo> current, std::shared_ptr<LayerInfo> next,
            std::shared_ptr<LayerInfo> next_next) {
        if (current->type != LayerType::LAYER_REDUCE_MEAN || next->type != LayerType::LAYER_REDUCE_MAX ||
                next_next->type != LayerType::LAYER_CONCAT) {
            return false;
        }
        auto reduce_mean_param = dynamic_cast<ReduceLayerParam *>(current->param.get());
        auto reduce_max_param = dynamic_cast<ReduceLayerParam *>(next->param.get());
        auto concat_param = dynamic_cast<ConcatLayerParam *>(next_next->param.get());
        if (!reduce_mean_param || !reduce_max_param || !concat_param) {
            return false;
        }
        if (concat_param->axis != 1 || (reduce_mean_param->axis.size() != 1 && reduce_mean_param->axis[0] != 1) ||
                (reduce_max_param->axis.size() != 1 && reduce_max_param->axis[0] != 1) ||
                current->inputs[0] != next->inputs[0] || current->outputs[0] != next_next->inputs[0] ||
                next->outputs[0] != next_next->inputs[1]) {
            return false;
        }
        return true;
    }

    Status NetOptimizerCbamFusedReduce::Optimize(NetStructure *structure, NetResource *resource) {
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

        int index = 0;
        for (; index < count - 2; index++) {
            auto layer_info_current = layers_orig[index];
            auto layer_info_next    = layers_orig[index + 1];
            auto layer_info_next_next = layers_orig[index + 2];
            if (NeedCbamReduceFusion(layer_info_current, layer_info_next, layer_info_next_next)) {
                std::shared_ptr<LayerInfo> layer_info_fused_reduce = std::make_shared<LayerInfo>();
                layer_info_fused_reduce->type = LayerType::LAYER_CBAM_FUSED_REDUCE;
                layer_info_fused_reduce->type_str = "CbamFusedReduce";
                layer_info_fused_reduce->name = layer_info_next_next->name;
                layer_info_fused_reduce->inputs = layer_info_current->inputs;
                layer_info_fused_reduce->outputs = layer_info_next_next->outputs;
                layer_info_fused_reduce->param = layer_info_next_next->param;
                layers_fused.push_back(layer_info_fused_reduce);
                index += 2;
            } else {
                layers_fused.push_back(layer_info_current);
            }
        }
        for (; index < count; index++) {
            layers_fused.push_back(layers_orig[index]);
        }
        structure->layers = layers_fused;

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS

