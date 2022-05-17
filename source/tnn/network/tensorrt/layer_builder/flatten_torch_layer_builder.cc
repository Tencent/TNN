// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(FlattenTorch, LAYER_FLATTENTORCH);

ILayer* FlattenTorchTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto layer_param = dynamic_cast<FlattenTorchLayerParam*>(param_);
    auto tensor = GetInputITensors()[0];

    int start_dim = layer_param->start_dim;
    int end_dim   = layer_param->end_dim;
    if (start_dim < 0) start_dim += tensor->getDimensions().nbDims;
    if (end_dim < 0) end_dim += tensor->getDimensions().nbDims;

    ShapeTensor in_dims = shapeOf(*tensor);
    ShapeTensor out_dims;
    if (start_dim > 0) {
        std::vector<int> d0_indices;
        for (int i=0; i<start_dim; i++) {
            d0_indices.push_back(i);
        }
        out_dims = gather(network, in_dims, ShapeTensor(1, std::move(d0_indices)));
        if (end_dim > start_dim) {
            ShapeTensor d1 = product(network, in_dims, start_dim, end_dim+1, 1);
            out_dims = concat(network, out_dims, d1);
        }
    } else {
        // Assume end_dim > start_dim when start_dim = 0
        out_dims = product(network, in_dims, 0, end_dim+1, 1);
    }
    if (end_dim < tensor->getDimensions().nbDims-1) {
        std::vector<int> d2_indices;
        for (int i=end_dim+1; i<tensor->getDimensions().nbDims; i++) {
            d2_indices.push_back(i);
        }
        ShapeTensor d2 = gather(network, in_dims, ShapeTensor(1, std::move(d2_indices)));
        out_dims = concat(network, out_dims, d2);
    }

    IShuffleLayer* flatten_layer = addShuffle(network, *tensor, out_dims, false);
    if (flatten_layer != nullptr) {
        flatten_layer->setName(layer_name_.c_str());
    }
    return flatten_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(FlattenTorch, LAYER_FLATTENTORCH);

}  //  namespace TNN_NS

