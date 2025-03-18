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

DECLARE_TENSORRT_LAYER_BUILDER(Floor, LAYER_FLOOR);

ILayer* FloorTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();

    if (tensor->getType()==nvinfer1::DataType::kINT32) {
        ILayer* identity_layer = network->addIdentity(*tensor);
        return identity_layer;
    }

    const auto input_dim = tensor->getDimensions().nbDims;
    IShuffleLayer *unsqueeze = nullptr, *squeeze = nullptr;
    if (input_dim == 0) {
        unsqueeze = addUnsqueeze(network, *tensor, {0,});
        if (unsqueeze == nullptr) {
            return unsqueeze;
        } else {
            unsqueeze->setName((layer_name_ + "/before_unsqueeze").c_str());
        }
        tensor = unsqueeze->getOutput(0);
    }
    IUnaryLayer* layer = network->addUnary(*tensor, UnaryOperation::kFLOOR);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }
    if (input_dim == 0) {
        squeeze = addSqueeze(network, *tensor, {0,});
        if (squeeze == nullptr) {
            return squeeze;
        } else {
            squeeze->setName((layer_name_ + "/after_squeeze").c_str());
        }
    }

    return input_dim > 0 ? static_cast<ILayer*>(layer) : static_cast<ILayer*>(squeeze);
}

REGISTER_TENSORRT_LAYER_BUILDER(Floor, LAYER_FLOOR);

}  //  namespace TNN_NS
