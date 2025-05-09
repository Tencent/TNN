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
    if (input_data_tensor->getType()==nvinfer1::DataType::kBOOL) {
        // Expand Slice Layer does not support format other than INT32
        // We need to turn BOOL, maybe output of EQUAL etc, to INT32 here. 
        ILayer* cast_layer = network->addIdentity(*input_data_tensor);
        cast_layer->setName((layer_name_+"_bool2int").c_str());
        cast_layer->setOutputType(0, nvinfer1::DataType::kINT32);
        input_data_tensor = cast_layer->getOutput(0);
    }

    ITensor* inputDims;
    if (input_tensors[0]->getDimensions().nbDims != 0)
        #if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 100
        inputDims = network->addCast(*(network->addShape(*input_tensors[0])->getOutput(0)), nvinfer1::DataType::kINT32)->getOutput(0);
        #else
        inputDims = network->addShape(*input_tensors[0])->getOutput(0);
        #endif
    int inputRank;
    if (input_tensors[0]->getDimensions().nbDims != 0) {
        inputRank = inputDims->getDimensions().d[0];
    } else {
        inputRank = 0;
    }

    nvinfer1::ITensor* shape;
    int shapeLength;
    if (input_tensors.size() == 2 && input_tensors[1]->getDimensions().d[0]!=-1) {
        shape = input_tensors[1];
        shapeLength = input_tensors[1]->getDimensions().d[0];
    } else if (input_tensors.size() == 1 || 
               (input_tensors.size() == 2 && input_tensors[1]->getDimensions().d[0]==-1)) {
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
        if (layer_param->shape.data() != nullptr) {
            tmpWeight.values = layer_param->shape.data();
        } else {
            // make sure tmpWeight.values is not nullptr
            tmpWeight.values = input_data_tensor;
        }
        tmpWeight.count = 1;
        if (input_tensors[0]->getDimensions().nbDims != 0) {
            #if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 100
            auto int64_shape_tensor = network->addShape(*network->addConstant(tmpDims, tmpWeight)->getOutput(0))->getOutput(0);
            nvinfer1::ITensor* const args[2] = {
                network->addCast(*int64_shape_tensor, nvinfer1::DataType::kINT32)->getOutput(0), inputDims};
            #else
            nvinfer1::ITensor* const args[2] = {
                network->addShape(*network->addConstant(tmpDims, tmpWeight)->getOutput(0))->getOutput(0), inputDims};
            #endif
            newDims = network->addConcatenation(args, 2)->getOutput(0);
        } else {
            #if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 100
            auto int64_shape_tensor = network->addShape(*network->addConstant(tmpDims, tmpWeight)->getOutput(0))->getOutput(0);
            newDims = network->addCast(*int64_shape_tensor, nvinfer1::DataType::kINT32)->getOutput(0);
            #else
            newDims = network->addShape(*network->addConstant(tmpDims, tmpWeight)->getOutput(0))->getOutput(0);
            #endif
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
        #if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 100
        nvinfer1::ITensor* const args[2] = {
            network->addCast(*(network->addShape(*network->addConstant(tmpDims, tmpWeight)->getOutput(0))->getOutput(0)),
                             nvinfer1::DataType::kINT32)->getOutput(0), shape};
        #else
        nvinfer1::ITensor* const args[2] = {
            network->addShape(*network->addConstant(tmpDims, tmpWeight)->getOutput(0))->getOutput(0), shape};
        #endif
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
        auto tmpValuePtr = std::make_shared<std::vector<int>>(1, 1);
        tmpWeight.values = tmpValuePtr.get(); 
        tmpWeight.count = 1;
        ILayer* one_shape_constant_layer = network->addConstant(tmpDims, tmpWeight);
        one_shape_constant_layer->setName((layer_name_+"_one_shape_constant").c_str());
        #if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 100
        one = network->addCast(*(network->addShape(*one_shape_constant_layer->getOutput(0))->getOutput(0)), nvinfer1::DataType::kINT32)->getOutput(0);
        #else
        one = network->addShape(*one_shape_constant_layer->getOutput(0))->getOutput(0);
        #endif
    }

    ITensor* strides = network->addElementWise(*one,
        *network->addElementWise(*newDims, *one, ElementWiseOperation::kSUB)->getOutput(0),
        ElementWiseOperation::kMIN)->getOutput(0);

    ISliceLayer* broadcast_layer = network->addSlice(*input_data_tensor, startDims, nvinfer1::Dims{}, nvinfer1::Dims{});
    broadcast_layer->setName((layer_name_+"_expand_slice").c_str());
    if (broadcast_layer != nullptr) {
        broadcast_layer->setInput(2, *sizes);
        broadcast_layer->setInput(3, *strides);
    }

    return broadcast_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Expand, LAYER_EXPAND);

}  //  namespace TNN_NS
