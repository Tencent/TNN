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

DECLARE_TENSORRT_LAYER_BUILDER(Range, LAYER_RANGE);

ILayer* RangeTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto input_tensors = GetInputITensors();
    const ShapeTensor start(*input_tensors[0]);
    const ShapeTensor limit(*input_tensors[1]);
    const ShapeTensor delta(*input_tensors[2]);

    ShapeTensor zero = shapeScalar(0);
    ShapeTensor numberOfElements = max(network, sub(network, zero,
        floorDiv(network, sub(network, start, limit), delta)), zero);
    IFillLayer* layer = addFill(network, convertTo1D(network, numberOfElements), FillOperation::kLINSPACE);
    if (start.allValuesKnown() && delta.allValuesKnown()) {
        layer->setAlpha(start[0]);
        layer->setBeta(delta[0]);
        layer->setOutputType(0, nvinfer1::DataType::kINT32);
    } else {
        layer->setInput(1, start.tensor(network));
        layer->setInput(2, convertTo1D(network, delta).tensor(network));
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Range, LAYER_RANGE);

}  //  namespace TNN_NS
