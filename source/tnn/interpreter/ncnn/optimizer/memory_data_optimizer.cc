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

#include "tnn/interpreter/ncnn/optimizer/ncnn_optimizer.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "tnn/core/common.h"
#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/ncnn/optimizer/ncnn_optimizer_manager.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"
#include "tnn/interpreter/raw_buffer.h"

namespace TNN_NS {

namespace ncnn {

    DECLARE_NCNN_OPTIMIZER(MemoryData);

    NCNNOptimizerRegister<MemoryDataOptimizer> g_ncnn_memory_data_optimizer;

    std::string MemoryDataOptimizer::Strategy() {
        return "NCNNOptimizerRemoveMemoryData";
    }

    std::set<LayerType> binary_op_sets = {
        LAYER_ADD,
        LAYER_SUB,
        LAYER_MUL,
        LAYER_DIV,
    };

    Status convert_const_to_weights(std::shared_ptr<LayerInfo> op, std::shared_ptr<LayerInfo> const_op,
                                    NetResource *net_resource) {
        // EltwiseLayerResource * ele_res = new EltwiseLayerResource();
        std::shared_ptr<EltwiseLayerResource> ele_res(new EltwiseLayerResource());

        auto it = std::find(op->inputs.begin(), op->inputs.end(), const_op->name);
        if (it != op->inputs.end()) {
            op->inputs.erase(it);
        } else {
            return Status(TNNERR_NET_ERR, "Error in convert_const_to_weights");
        }

        ConstLayerParam *const_param = dynamic_cast<ConstLayerParam *>(const_op->param.get());

        if (const_param == nullptr) {
            return Status(TNNERR_NET_ERR, "Error: const param null.");
        }

        RawBuffer weights;
        if (net_resource->resource_map.count(const_op->name) > 0) {
            ConstLayerResource *const_res =
                dynamic_cast<ConstLayerResource *>(net_resource->resource_map[const_op->name].get());

            if (const_res == nullptr) {
                return Status(TNNERR_NET_ERR, "Error: const weights null.");
            }

            weights = const_res->weight_handle;
        } else {
#ifdef GENERATE_RESOURCE
            // generate weights in benchmark mode
            int weight_size = 1;
            for (auto dim_i : const_param->dims) {
                weight_size *= dim_i;
            }
            weights = RawBuffer(weight_size * sizeof(float));
            for (int i = 0; i < weight_size; i++) {
                weights.force_to<float *>()[i] = 1.0;
            }
#else
            return Status(TNNERR_NET_ERR, "Error: not found const weights.");
#endif
        }

        ele_res->element_handle = weights;
        ele_res->element_shape  = const_param->dims;

        net_resource->resource_map[op->name] = std::dynamic_pointer_cast<LayerResource>(ele_res);

        return TNN_OK;
    }

    Status MemoryDataOptimizer::Optimize(NetStructure *structure, NetResource *resource) {
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
        std::map<std::string, std::shared_ptr<LayerInfo>> const_layers;

        for (int index = 0; index < count; index++) {
            auto cur_layer = layers_orig[index];

            // mark Const Layer, aka MemoryData layer
            if (cur_layer->type == LAYER_CONST) {
                const_layers[cur_layer->name] = cur_layer;
                continue;
            }

            layers_fused.push_back(cur_layer);

            // only work for layer with 2 inputs
            if (cur_layer->inputs.size() != 2) {
                continue;
            }

            // only work for binary ops
            if (binary_op_sets.find(cur_layer->type) == binary_op_sets.end()) {
                continue;
            }

            // ignore this case: Const as the first input of LAYER_DIV
            if (cur_layer->type == LAYER_DIV && const_layers.find(cur_layer->inputs[0]) != const_layers.end()) {
                continue;
            }

            for (auto in_name : cur_layer->inputs) {
                if (const_layers.find(in_name) != const_layers.end()) {
                    auto status = convert_const_to_weights(cur_layer, const_layers[in_name], resource);
                    if (status != TNN_OK) {
                        return status;
                    }
                    const_layers.erase(in_name);
                }
            }
        }

        if (const_layers.size() != 0) {
            return Status(TNNERR_NET_ERR, "Error: unfused const layer.");
        }

        structure->layers = layers_fused;

        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
