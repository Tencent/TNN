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

DECLARE_TENSORRT_LAYER_BUILDER(Expand, LAYER_EXPAND);

ILayer* ExpandTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto layer_param = dynamic_cast<ExpandLayerParam*>(param_);

    auto input_tensors = GetInputITensors();
    ITensor* input_data_tensor = input_tensors[0];
    ITensor* inputDims;
    if (input_tensors[0]->getDimensions().nbDims != 0)
        inputDims = network->addShape(*input_tensors[0])->getOutput(0);
    int inputRank;
    if (input_tensors[0]->getDimensions().nbDims != 0) {
        inputRank = inputDims->getDimensions().d[0];
    } else {
        inputRank = 0;
    }

    nvinfer1::ITensor* shape;
    int shapeLength;
    if (input_tensors.size() == 2) {
        shape = input_tensors[1];
        shapeLength = input_tensors[1]->getDimensions().d[0];
    } else if (input_tensors.size() == 1) {
        nvinfer1::Dims shapeDims;
        shapeDims.nbDims = 1;
        shapeDims.d[0] = layer_param->shape.size();
        Weights shapeWeight;
        shapeWeight.type = nvinfer1::DataType::kINT32;
        shapeWeight.values = layer_param->shape.data();
        shapeWeight.count = layer_param->shape.size();
        auto shapeLayer = network->addConstant(shapeDims, shapeWeight);
        shape = shapeLayer->getOutput(0);
        shapeLength = layer_param->shape.size();
    }
    int newRank = std::max(shapeLength, inputRank);

    ITensor* newDims;
    if (newRank - inputRank != 0) {
        Dims tmpDims;
        tmpDims.nbDims = newRank - inputRank;
        for (int i = 0; i < newRank - inputRank; i++) {
            tmpDims.d[i] = 1;
        }
        Weights tmpWeight;
        tmpWeight.type = nvinfer1::DataType::kINT32;
        tmpWeight.values = layer_param->shape.data();
        tmpWeight.count = 1;
        if (input_tensors[0]->getDimensions().nbDims != 0) {
            nvinfer1::ITensor* const args[2] = {
                network->addShape(*network->addConstant(tmpDims, tmpWeight)->getOutput(0))->getOutput(0), inputDims};
            newDims = network->addConcatenation(args, 2)->getOutput(0);
        } else {
            newDims = network->addShape(*network->addConstant(tmpDims, tmpWeight)->getOutput(0))->getOutput(0);
        }
    } else {
        newDims = inputDims;
    }

    if (newRank - inputRank != 0) {
        IShuffleLayer* reshape_layer = network->addShuffle(*input_data_tensor);
        reshape_layer->setInput(1, *newDims);
        input_data_tensor = reshape_layer->getOutput(0);
    }

    ITensor* newShape;
    if (newRank - shapeLength != 0) {
        Dims tmpDims;
        tmpDims.nbDims = newRank - shapeLength;
        for (int i = 0; i < newRank - shapeLength; i++) {
            tmpDims.d[i] = 1;
        }
        Weights tmpWeight;
        tmpWeight.type = nvinfer1::DataType::kINT32;
        tmpWeight.values = layer_param->shape.data();
        tmpWeight.count = 1;
        nvinfer1::ITensor* const args[2] = {
            network->addShape(*network->addConstant(tmpDims, tmpWeight)->getOutput(0))->getOutput(0), shape};
        newShape = network->addConcatenation(args, 2)->getOutput(0);
    } else {
        newShape = shape;
    }

    Dims startDims;
    startDims.nbDims = newRank;
    for (int i = 0; i < newRank; i++) {
        startDims.d[i] = 0;
    }

    ITensor* sizes = network->addElementWise(*newDims, *newShape, ElementWiseOperation::kMAX)->getOutput(0);
    ITensor* one;
    {
        Dims tmpDims;
        tmpDims.nbDims = newRank;
        for (int i = 0; i < newRank; i++)
            tmpDims.d[i] = 1;
        Weights tmpWeight;
        tmpWeight.type = nvinfer1::DataType::kINT32;
        tmpWeight.values = layer_param->shape.data();
        tmpWeight.count = 1;
        one = network->addShape(*network->addConstant(tmpDims, tmpWeight)->getOutput(0))->getOutput(0);
    }

    ITensor* strides = network->addElementWise(*one,
        *network->addElementWise(*newDims, *one, ElementWiseOperation::kSUB)->getOutput(0),
        ElementWiseOperation::kMIN)->getOutput(0);

    ISliceLayer* broadcast_layer = network->addSlice(*input_data_tensor, startDims, nvinfer1::Dims{}, nvinfer1::Dims{});
    if (broadcast_layer != nullptr) {
        broadcast_layer->setInput(2, *sizes);
        broadcast_layer->setInput(3, *strides);
    }

    return broadcast_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Expand, LAYER_EXPAND);

}  //  namespace TNN_NS
