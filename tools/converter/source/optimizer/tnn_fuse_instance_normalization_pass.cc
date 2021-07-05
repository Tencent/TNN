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

#include "tnn_optimize_pass.h"
namespace TNN_CONVERTER {

DECLARE_OPTIMIZE_PASS(FuseInstanceNormalization);

std::string TnnOptimizeFuseInstanceNormalizationPass::PassName() {
    return "FuseInstanceNormalization";
}

TNN_NS::Status TnnOptimizeFuseInstanceNormalizationPass::exec(TNN_NS::NetStructure& net_structure,
                                                              TNN_NS::NetResource& net_resource) {
    auto& layers = net_structure.layers;
    for (auto iter = layers.begin(); iter != layers.end(); iter++) {
        auto& layer = *iter;
        if (layer->type != TNN_NS::LAYER_REDUCE_MEAN) {
            continue;
        }
        auto reduce_mean_param = dynamic_cast<TNN_NS::ReduceLayerParam*>(layer->param.get());
        // take care belpw "!"
        if (!(reduce_mean_param->axis.size() == 2 && reduce_mean_param->axis[0] == 2 &&
              reduce_mean_param->axis[1] == 3)) {
            continue;
        }
        auto pooling_layer_param       = new TNN_NS::PoolingLayerParam;
        pooling_layer_param->type      = "Pooling";
        pooling_layer_param->name      = reduce_mean_param->name;
        pooling_layer_param->quantized = reduce_mean_param->quantized;
        // pool_type 1 meaning: AveragePool
        pooling_layer_param->pool_type      = 1;
        pooling_layer_param->kernels        = {0, 0};
        pooling_layer_param->kernels_params = pooling_layer_param->kernels;
        pooling_layer_param->strides        = {1, 1};
        pooling_layer_param->pads           = {0, 0, 0, 0};
        pooling_layer_param->kernel_indexs  = {-1, -1};
        pooling_layer_param->pad_type       = -1;
        pooling_layer_param->ceil_mode      = 0;
        // update layer
        layer->type     = TNN_NS::LAYER_POOLING;
        layer->type_str = "Pooling";
        layer->param    = std::shared_ptr<TNN_NS::LayerParam>(pooling_layer_param);
    }
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_OPTIMIZE_PASS(FuseInstanceNormalization);
}  // namespace TNN_CONVERTER
