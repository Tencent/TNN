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

#include "tnn/utils/dims_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"
#include "tnn/network/tensorrt/utils.h"


namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(Pow, LAYER_POWER);

ILayer* PowTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto params = dynamic_cast<PowLayerParam *>(param_);

    auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
    auto input_dims        = input_blobs_[0]->GetBlobDesc().dims;

    if (params->exponent == 0.5 && params->shift == 0 && params->scale == 1) {
        if (tensor->getDimensions().nbDims == 0) {
            IShuffleLayer* shuffle_layer = network->addShuffle(*GetInputITensors()[0]);
            nvinfer1::Dims d;
            d.nbDims = 1;
            d.d[0] = 1;
            shuffle_layer->setReshapeDimensions(d);
            tensor = shuffle_layer->getOutput(0);
        }
        IUnaryLayer* layer = network->addUnary(*tensor, UnaryOperation::kSQRT);
        if (layer != nullptr) {
            layer->setName(layer_name_.c_str());
        }
        return layer;
    }

    Weights power;
    power.type = nvinfer1::DataType::kFLOAT;
    power.count = 1;
    power.values = &(params->exponent);

    Weights shift;
    shift.type = nvinfer1::DataType::kFLOAT;
    shift.count = 1;
    shift.values = &(params->shift);

    Weights scale;
    scale.type = nvinfer1::DataType::kFLOAT;
    scale.count = 1;
    scale.values = &(params->scale);

    int dims_size = tensor->getDimensions().nbDims;
    // unsqueeze 
    ILayer* layer;
    if (dims_size == 2 || dims_size == 3) {
        DimsVector unsqueeze_dims;
        for (int i = 0; i < dims_size; i++) {
            unsqueeze_dims.push_back(0);
        }
        for (int i = 0; i < 4 - dims_size; i++) {
            unsqueeze_dims.push_back(1);
        }
        layer = AddReshapeToNetwork(network, tensor, unsqueeze_dims, (layer_name_ + "squeeze").c_str());
        tensor = layer->getOutput(0);
    }

    layer = network->addScaleNd(*tensor, ScaleMode::kUNIFORM, shift, scale, power, 1);

    if (layer != NULL) {
        layer->setName(layer_name_.c_str());
        tensor = layer->getOutput(0);
    }

    //squeeze
    if(dims_size == 2 || dims_size == 3) {
        DimsVector squeeze_dims;
        for (int i = 0; i < dims_size; i++) {
            squeeze_dims.push_back(0);
        }
        layer = AddReshapeToNetwork(network, tensor, squeeze_dims, (layer_name_ + "unsqueeze").c_str());
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Pow, LAYER_POWER);

}  //  namespace TNN_NS


