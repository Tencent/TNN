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

    DECLARE_NCNN_OPTIMIZER(ExpandSlice);

    NCNNOptimizerRegister<ExpandSliceOptimizer> g_ncnn_expand_slice_optimizer;

    std::string ExpandSliceOptimizer::Strategy() {
        return "NCNNOptimizerExpandSlice";
    }

    Status expand_slice(std::shared_ptr<LayerInfo> layer, std::vector<std::shared_ptr<LayerInfo>> &output_layers) {
        output_layers.resize(0);

        SliceLayerParam *slice_param = dynamic_cast<SliceLayerParam *>(layer->param.get());
        if (slice_param == NULL) {
            return Status(TNNERR_NET_ERR, "Error: slice param nil.");
        }

        if (layer->outputs.size() != slice_param->slices.size()) {
            return Status(TNNERR_NET_ERR, "Error: slice param error.");
        }

        std::vector<int> ones  = {1, 1, 1, 1};
        std::vector<int> zeros = {0, 0, 0, 0};

        int begin_acc = 0;
        for (size_t i = 0; i < layer->outputs.size(); i++) {
            auto out_name = layer->outputs[i];

            LayerInfo *new_layer = new LayerInfo();
            new_layer->name      = out_name;
            new_layer->type      = LAYER_STRIDED_SLICE;
            new_layer->type_str  = "StridedSlice";
            new_layer->inputs    = layer->inputs;
            new_layer->outputs   = {out_name};

            StrideSliceLayerParam *new_param = new StrideSliceLayerParam();
            new_param->strides               = ones;
            new_param->begins                = zeros;
            new_param->ends                  = zeros;
            int slice_size                   = slice_param->slices[i];
            // order [w h d c n]
            new_param->begins[3 - slice_param->axis] = begin_acc;
            new_param->ends[3 - slice_param->axis]   = begin_acc + slice_size;

            new_layer->param = std::shared_ptr<LayerParam>(new_param);
            output_layers.push_back(std::shared_ptr<LayerInfo>(new_layer));

            if (slice_size == -233) {
                new_param->ends[3 - slice_param->axis] = 0;
                break;
            }
            begin_acc += slice_size;
        }

        if (output_layers.size() != layer->outputs.size()) {
            return Status(TNNERR_NET_ERR, "Error: expand slice fail.");
        }
        return TNN_OK;
    }

    Status ExpandSliceOptimizer::Optimize(NetStructure *structure, NetResource *resource) {
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

        for (int index = 0; index < count; index++) {
            auto cur_layer = layers_orig[index];
            if (cur_layer->type == LAYER_SLICE) {
                std::vector<std::shared_ptr<LayerInfo>> expanded_layers;
                RETURN_ON_NEQ(expand_slice(cur_layer, expanded_layers), TNN_OK);
                layers_fused.insert(layers_fused.end(), expanded_layers.begin(), expanded_layers.end());
            } else {
                layers_fused.push_back(cur_layer);
            }
        }

        structure->layers = layers_fused;

        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
