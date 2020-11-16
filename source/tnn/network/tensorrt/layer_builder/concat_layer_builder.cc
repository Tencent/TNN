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

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(Concat, LAYER_CONCAT);

ILayer* ConcatTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<ConcatLayerParam*>(param_);
    size_t nbInputs = input_blobs_.size();
    ITensor * input_tensors[nbInputs];
    for (int i = 0; i < nbInputs; i++) {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[i])->GetForeignTensor();
        auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
        input_tensors[i] = tensor;
    }

    IConcatenationLayer* layer = network->addConcatenation(input_tensors, nbInputs);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
        layer->setAxis(paramlist->axis);
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Concat, LAYER_CONCAT);

}  //  namespace TNN_NS
