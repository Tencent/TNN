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

DECLARE_TENSORRT_LAYER_BUILDER(Where, LAYER_WHERE);

ILayer* WhereTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    ITensor *x, *y, *condition;
    auto input_tensors = GetInputITensors();
    if (input_tensors.size()==3) {
        x = input_tensors[0];
        y = input_tensors[1];
        condition = input_tensors[2];
    } else {
        auto layer_resource = dynamic_cast<WhereLayerResource*>(resource_);
        if (!layer_resource) {
            LOGE("WhereTRTLayerBuilder: Unable to Get LayerResource while at least one of x, y missing.");
            return nullptr;
        }
 
        if (layer_resource->x.GetBytesSize()>0 && layer_resource->y.GetBytesSize()>0) {
            auto x_const_layer = ConvertWeightToConstLayer(network, &(layer_resource->x));
            auto y_const_layer = ConvertWeightToConstLayer(network, &(layer_resource->y));
            if (x_const_layer==nullptr || y_const_layer==nullptr) {
                LOGE("WhereTRTLayerBuilder: Unable to to turn x or y in LayerResource to TRT constant layer.");
                return nullptr;
            }
            x = x_const_layer->getOutput(0);
            y = y_const_layer->getOutput(0);
            condition = input_tensors[0];
        } else if (layer_resource->x.GetBytesSize()>0) {
            auto x_const_layer = ConvertWeightToConstLayer(network, &(layer_resource->x));
            if (x_const_layer==nullptr) {
                LOGE("WhereTRTLayerBuilder: Unable to to turn x in LayerResource to TRT constant layer.");
                return nullptr;
            }
            x = x_const_layer->getOutput(0);
            y = input_tensors[0];
            condition = input_tensors[1];
        } else if (layer_resource->y.GetBytesSize()>0) {
            auto y_const_layer = ConvertWeightToConstLayer(network, &(layer_resource->y));
            if (y_const_layer==nullptr) {
                LOGE("WhereTRTLayerBuilder: Unable to to turn x in LayerResource to TRT constant layer.");
                return nullptr;
            }
            x = input_tensors[0];
            y = y_const_layer->getOutput(0);
            condition = input_tensors[1];
        } else {
            LOGE("WhereTRTLayerBuilder: Unable to Get LayerResource while at least one of x, y missing.");
            return nullptr;
        }
    }

    if (condition->getType() == nvinfer1::DataType::kFLOAT  ||
        condition->getType() == nvinfer1::DataType::kHALF ||
        condition->getType() == nvinfer1::DataType::kINT32) {
        ILayer* cast_layer = network->addIdentity(*condition);
        cast_layer->setOutputType(0, nvinfer1::DataType::kBOOL);
        condition = cast_layer->getOutput(0);
    }

    // aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value)
    // when %self is float tensor, %value may be zero tensor of int type
    if (y->getType() != x->getType()) {
        // cast x to the same type of y
        ILayer* cast_layer = network->addIdentity(*x);
        cast_layer->setOutputType(0, y->getType());
        x = cast_layer->getOutput(0);
    }

    BroadcastTensors(network, x, y, condition);

    ISelectLayer* layer = network->addSelect(*condition, *x, *y);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }
    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Where, LAYER_WHERE);

}  //  namespace TNN_NS
