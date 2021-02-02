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

#include <vector>
#include <algorithm>

#include "tnn/network/tensorrt/layer_builder/reduce_layer_builder.h"

namespace TNN_NS {

ReduceTRTLayerBuilder::ReduceTRTLayerBuilder(LayerType ignore) : TensorRTLayerBuilder(ignore) {
}

ILayer* ReduceTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<ReduceLayerParam*>(param_);
    auto axis = paramlist->axis;
    uint32_t reduceAxis = 0x0;
    if (std::find(axis.begin(), axis.end(), 1) != axis.end()) {
        reduceAxis |= 0x2;
    }
    if (std::find(axis.begin(), axis.end(), 2) != axis.end()) {
        reduceAxis |= 0x4;
    }
    if (std::find(axis.begin(), axis.end(), 3) != axis.end()) {
        reduceAxis |= 0x8;
    }

    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
    IReduceLayer* layer = network->addReduce(*tensor, m_op, reduceAxis, paramlist->keep_dims == 1);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }
    
    return layer;
}

}  //  namespace TNN_NS
