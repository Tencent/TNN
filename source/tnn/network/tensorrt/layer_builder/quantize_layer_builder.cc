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

#include "tnn/network/tensorrt/tensorrt_network.h"
#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"
#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(Quantize, LAYER_QUANTIZE);

ILayer* QuantizeTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
#if NV_TENSORRT_MAJOR < 8
    LOGE("quant layer builder is not support before TensorRT8\n");
    return nullptr;
#else
    auto layer_param = dynamic_cast<QuantizeLayerParam*>(param_);
    auto tensor = GetInputITensors()[0];
    //auto scale = GetInputITensors()[1];
    auto layer_resource = dynamic_cast<QuantizeLayerResource*>(resource_);
    auto const_layer = ConvertWeightToConstLayer(network, &(layer_resource->scale_handle));
    nvinfer1::ITensor * scale = nullptr;
    if (const_layer != nullptr) {
        scale = const_layer->getOutput(0);
    }

    int64_t axis = layer_param->axis;


    IQuantizeLayer* quantize_layer = network->addQuantize(*tensor, *scale);
    quantize_layer->setAxis(axis);
    if (quantize_layer != nullptr) {
        quantize_layer->setName(layer_name_.c_str());
    }
    return quantize_layer;
#endif
}

REGISTER_TENSORRT_LAYER_BUILDER(Quantize, LAYER_QUANTIZE);

}  //  namespace TNN_NS

