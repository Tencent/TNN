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

DECLARE_TENSORRT_LAYER_BUILDER(Linspace, LAYER_LINSPACE);

ILayer* LinspaceTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto layer_param = dynamic_cast<LinspaceLayerParam*>(param_);
    auto input_tensors = GetInputITensors();
    ShapeTensor start;
    ShapeTensor start_vec;
    ShapeTensor end;
    ShapeTensor steps;
    if (input_tensors.size() == 3) {
        start = ShapeTensor(*input_tensors[0]);
        end = ShapeTensor(*input_tensors[1]);
        steps = ShapeTensor(*input_tensors[2]);
    } else { // input_tensors.size() < 3
        if (layer_param->start_index == -1) {
            start = shapeScalar(static_cast<int>(layer_param->start.f));
            start_vec = shapeVector(static_cast<int>(layer_param->start.f));
        } else {
            start = ShapeTensor(*input_tensors[layer_param->start_index]);
        }
        if (layer_param->end_index==-1) {
            end = shapeVector(static_cast<int>(layer_param->end.f));
        } else {
            end = ShapeTensor(*input_tensors[layer_param->end_index]);
        }
        if (layer_param->steps_index == -1) {
            steps = shapeVector(layer_param->steps.i);
        } else {
            steps = ShapeTensor(*input_tensors[layer_param->steps_index]);
        }
    }

    ShapeTensor zero;
    if (start_vec.rank() == 0) {
        zero = shapeScalar(0);
    } else {
        zero = shapeVector(0);
    }
    if (steps.rank() == 0) {
        steps = convertTo1D(network, steps);
    }
    ShapeTensor step1 = sub(network, start_vec, end);
    ShapeTensor step2 = floorDiv(network, step1, steps);
    ShapeTensor step3 = sub(network, zero, step2);
    IFillLayer* layer = addFill(network, steps, FillOperation::kLINSPACE);
    if (start.allValuesKnown() && end.allValuesKnown()) {
        layer->setAlpha(layer_param->start.f);
        layer->setBeta(layer_param->end.f);
        layer->setOutputType(0, nvinfer1::DataType::kINT32);
    } else {
        layer->setInput(1, start.tensor(network));
        layer->setInput(2, convertTo1D(network, step3).tensor(network));
        layer->setOutputType(0, nvinfer1::DataType::kINT32);
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Linspace, LAYER_LINSPACE);

}  //  namespace TNN_NS
