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

#include "tnn/optimizer/net_optimizer_remove_inplace_ops.h"

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "tnn/core/common.h"
#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"

namespace TNN_NS {

namespace optimizer {

    // P1 priority: should be fuse after bn scale fuse
    NetOptimizerRegister<NetOptimizerRemoveInplaceOps> g_net_optimizer_remove_inplace_ops(OptPriority::P1);

    std::string NetOptimizerRemoveInplaceOps::Strategy() {
        return kNetOptimizerRemoveInplaceOps;
    }

    bool NetOptimizerRemoveInplaceOps::IsSupported(const NetworkConfig &net_config) {
        return true;
    }

    Status NetOptimizerRemoveInplaceOps::Optimize(NetStructure *structure, NetResource *resource) {
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

        std::map<std::string, std::string> rename_map;

        for (int index = 0; index < count; index++) {
            auto layer = layers_orig[index];
            if (layer->type == LAYER_INPLACE_COPY && layer->inputs.size() == 2) {
                auto in_name0 = layer->inputs[0];
                auto in_name1 = layer->inputs[1];

                bool is_slice_copy_pattern = false;
                // aten::slice + aten::copy_
                std::string slice_in_name, new_slice_in_name;
                std::string slice_layer_name;
                std::shared_ptr<LayerInfo> slice_layer;
                for (int slice_index = 0; slice_index < index; slice_index++) {
                    slice_layer = layers_orig[slice_index];
                    for (auto out_name : slice_layer->outputs) {
                        if (out_name == in_name0 && slice_layer->type == LAYER_STRIDED_SLICE_V2) {
                            slice_layer_name = slice_layer->name;
                            slice_in_name = slice_layer->inputs[0];
                            new_slice_in_name = slice_in_name + "_inplace";
                            rename_map[slice_in_name] = new_slice_in_name;
                            rename_map[out_name] = in_name1;
                            is_slice_copy_pattern = true;
                            break;
                        }
                    }
                }
                for (auto out_name : layer->outputs) {
                    if (rename_map.find(out_name) == rename_map.end()) {
                        rename_map[out_name] = is_slice_copy_pattern ? in_name1 : in_name0;
                    } else {
                        return Status(TNNERR_NET_ERR, "duplicated output blobs");
                    }
                }
                if (is_slice_copy_pattern) {
                    layers_fused.erase(std::remove_if(layers_fused.begin(), layers_fused.end(),
                                                      [slice_layer_name](const std::shared_ptr<LayerInfo>& layer) {
                                                            return layer->name == slice_layer_name;
                                                        }),
                                       layers_fused.end());
                    std::shared_ptr<LayerInfo> new_layer = std::make_shared<LayerInfo>();
                    new_layer->type = LAYER_INPLACE_SLICE_COPY;
                    new_layer->type_str = "InplaceSliceCopy";
                    new_layer->name = new_slice_in_name;
                    new_layer->inputs = {slice_in_name, in_name1};
                    new_layer->outputs = {new_slice_in_name};
                    new_layer->param = slice_layer->param;
                    layers_fused.push_back(new_layer);
                    structure->blobs.insert(new_slice_in_name);
                }
            } else {
                std::vector<std::string> new_inputs;
                new_inputs.reserve(layer->inputs.size());
                for (auto in_name : layer->inputs) {
                    while (rename_map.find(in_name) != rename_map.end()) {
                        in_name = rename_map[in_name];
                    }
                    new_inputs.push_back(in_name);
                }
                layer->inputs = new_inputs;
                layers_fused.push_back(layer);
            }
        }

        structure->layers = layers_fused;
        return TNN_OK;

    }

}  // namespace optimizer

}  // namespace TNN_NS
