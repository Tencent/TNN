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

    int dims_size = input_tensor->getDimensions().nbDims;

    //unsqueeze
    if (input_tensor->getDimensions().nbDims < 4) {
        DimsVector unsqueeze_dims;
        for (int i = 0; i < dims_size; i++) {
            unsqueeze_dims.push_back(0);
        }
        while(unsqueeze_dims.size() < 4) {
            unsqueeze_dims.push_back(1);
        }
        layer = AddReshapeToNetwork(network, input_tensor, unsqueeze_dims, (layer_name_ + "unsqueeze").c_str());
        input_tensor = layer->getOutput(0);
    }

    ISoftMaxLayer* softmax_layer = network->addSoftMax(*input_tensor);
    if (softmax_layer != nullptr) {
        softmax_layer->setName(layer_name_.c_str());
        softmax_layer->setAxes(1 << paramlist->axis);
        input_tensor = softmax_layer->getOutput(0);
        layer = softmax_layer;
    }

    auto output_dims = output_blobs_[0]->GetBlobDesc().dims;
    //squeeze
    if(dims_size < 4) {
        DimsVector squeeze_dims;
        for (int i = 0; i < dims_size; i++) {
            squeeze_dims.push_back(0);
        }
        layer = AddReshapeToNetwork(network, input_tensor, squeeze_dims, (layer_name_ + "squeeze").c_str());
    }
    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Softmax, LAYER_SOFTMAX);

}  //  namespace TNN_NS
