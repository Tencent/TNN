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

DECLARE_TENSORRT_LAYER_BUILDER(Neg, LAYER_NEG);

ILayer* NegTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();

    if (tensor->getDimensions().nbDims == 0) {
        IShuffleLayer* shuffle_layer = network->addShuffle(*GetInputITensors()[0]);
        nvinfer1::Dims d;
        d.nbDims = 1;
        d.d[0] = 1;
        shuffle_layer->setReshapeDimensions(d);
        tensor = shuffle_layer->getOutput(0);
    }

    ILayer* layer;    
    nvinfer1::DataType in_dtype = tensor->getType();
    // TRT8 unary NEG does not suppport INT32
    // Convert to FLOAT first and then back to INT32 AFTER ABS 
    if (in_dtype==nvinfer1::DataType::kINT8 || in_dtype==nvinfer1::DataType::kINT32) {
        ILayer* cast_layer = network->addIdentity(*tensor);
        cast_layer->setName((layer_name_+"_int2fp").c_str());
        cast_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        tensor = cast_layer->getOutput(0);
    }

    // Main Neg OP
    layer = network->addUnary(*tensor, UnaryOperation::kNEG);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }
 
    // Cast back to original data type
    if (in_dtype==nvinfer1::DataType::kINT8 || in_dtype==nvinfer1::DataType::kINT32) {
        layer = network->addIdentity(*tensor);
        layer->setName((layer_name_+"_fp2int").c_str());
        layer->setOutputType(0, in_dtype);
        tensor = layer->getOutput(0);
    }

    if (GetInputITensors()[0]->getDimensions().nbDims == 0) {
        IShuffleLayer* shuffle_layer = network->addShuffle(*layer->getOutput(0));
        nvinfer1::Dims d;
        d.nbDims = 0;
        shuffle_layer->setReshapeDimensions(d);
        return shuffle_layer;
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Neg, LAYER_NEG);

}  //  namespace TNN_NS
