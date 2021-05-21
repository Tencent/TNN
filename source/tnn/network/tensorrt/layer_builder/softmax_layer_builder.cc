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

DECLARE_TENSORRT_LAYER_BUILDER(Softmax, LAYER_SOFTMAX);

ILayer* SoftmaxTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<SoftmaxLayerParam*>(param_);
    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
    ILayer* layer;

    ISoftMaxLayer* softmax_layer = network->addSoftMax(*input_tensor);
    if (softmax_layer != nullptr) {
        softmax_layer->setName(layer_name_.c_str());
        softmax_layer->setAxes(1 << paramlist->axis);
        input_tensor = softmax_layer->getOutput(0);
        layer = softmax_layer;
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Softmax, LAYER_SOFTMAX);

}  //  namespace TNN_NS
