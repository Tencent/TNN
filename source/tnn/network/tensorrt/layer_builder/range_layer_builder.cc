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
    auto layer_param = dynamic_cast<RangeLayerParam*>(param_);
    auto input_tensors = GetInputITensors();
    ShapeTensor start;
    ShapeTensor limit;
    ShapeTensor delta;
    if (input_tensors.size()==3) {
        start = ShapeTensor(*input_tensors[0]);
        limit = ShapeTensor(*input_tensors[1]);
        delta = ShapeTensor(*input_tensors[2]);
    } else { //input_tensors.size()<3
        if (layer_param->start_index==-1) {
            //std::cout << "[Range AddToNet] start_index, start.i = " << layer_param->start.i << std::endl;
            start = shapeVector(layer_param->start.i);
        } else {
            //std::cout << "[Range AddToNet] start_index, param_idx = " << layer_param->start_index << std::endl;
            start = ShapeTensor(*input_tensors[layer_param->start_index]);
        }
        if (layer_param->limit_index==-1) {
            //std::cout << "[Range AddToNet] limit_index, limit.i = " << layer_param->limit.i << std::endl;
            limit = shapeVector(layer_param->limit.i);
        } else {
            //std::cout << "[Range AddToNet] limit_index, param_idx = " << layer_param->limit_index << std::endl;
            limit = ShapeTensor(*input_tensors[layer_param->limit_index]);
            //std::cout << "[Range AddToNet] limit.allValuesKnown = " << (int)limit.allValuesKnown() << std::endl;
            //std::cout << "[Range AddToNet] limit[0] = " << (int)limit[0] << std::endl;
        }
        if (layer_param->delta_index==-1) {
            //std::cout << "[Range AddToNet] delta_index, delta.i = " << layer_param->delta.i << std::endl;
            delta = shapeVector(layer_param->delta.i);
        } else {
            //std::cout << "[Range AddToNet] delta_index, param_idx = " << layer_param->delta_index << std::endl;
            delta = ShapeTensor(*input_tensors[layer_param->delta_index]);
        }
    }

    //ShapeTensor zero = shapeScalar(0);
    ShapeTensor zero;
    if (start.rank()==0) {
        zero = shapeScalar(0);
    } else {
        zero = shapeVector(0);
    }
    ShapeTensor step1 = sub(network, start, limit);
    ShapeTensor step2 = floorDiv(network, step1, delta);
    ShapeTensor step3 = sub(network, zero, step2);
    ShapeTensor numberOfElements = max(network, step3, zero);
    if (numberOfElements.rank()==0) {
        numberOfElements = convertTo1D(network, numberOfElements);
    }
    IFillLayer* layer = addFill(network, numberOfElements, FillOperation::kLINSPACE);
    if (start.allValuesKnown() && delta.allValuesKnown()) {
        layer->setAlpha(start[0]);
        layer->setBeta(delta[0]);
        layer->setOutputType(0, nvinfer1::DataType::kINT32);
    } else {
        layer->setInput(1, start.tensor(network));
        layer->setInput(2, convertTo1D(network, delta).tensor(network));
        layer->setOutputType(0, nvinfer1::DataType::kINT32);
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Range, LAYER_RANGE);

}  //  namespace TNN_NS
