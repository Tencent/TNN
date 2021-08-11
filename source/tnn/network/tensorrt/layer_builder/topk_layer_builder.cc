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

DECLARE_TENSORRT_LAYER_BUILDER(TopK, LAYER_TOPK);

ILayer* TopKTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto param = dynamic_cast<TopKLayerParam*>(param_);
    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();

    auto topk_largest = nvinfer1::TopKOperation::kMAX;
    if (param->largest != 1) {
        topk_largest = nvinfer1::TopKOperation::kMIN;
    }

    uint32_t reduceAxis = 0x1 << param->axis;

    ITopKLayer* layer = network->addTopK(*tensor, topk_largest, param->k, reduceAxis);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(TopK, LAYER_TOPK);

}  //  namespace TNN_NS