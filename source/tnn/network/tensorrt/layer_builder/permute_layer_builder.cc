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

#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"
#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(Permute, LAYER_PERMUTE);

ILayer* PermuteTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<PermuteLayerParam*>(param_);
    Permutation permute;
    for (int i = 0; i < paramlist->orders.size(); ++i) {
        permute.order[i] = paramlist->orders[i];
    }

    Blob* input_blob  = input_blobs_[0];
    auto input_tensors = GetInputITensors();
    IShuffleLayer* layer = network->addShuffle(*input_tensors[0]);
    if (layer != nullptr) {
        Dims reshape_dims;
        reshape_dims.nbDims = input_tensors[0]->getDimensions().nbDims;
        for (int i = 0; i < reshape_dims.nbDims; i++) {
            reshape_dims.d[i] = 0;
        }
        layer->setName(layer_name_.c_str());
        layer->setReshapeDimensions(reshape_dims);
        layer->setSecondTranspose(permute);
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Permute, LAYER_PERMUTE);

}  //  namespace TNN_NS

