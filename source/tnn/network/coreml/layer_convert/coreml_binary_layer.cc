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

#include "coreml_base_layer.h"
#include "coreml_const_layer.h"
#include "coreml_binary_layer.h"

namespace TNN_NS {

Status CoreMLBinaryLayer::BuildLayerType() {
    //layer type
    return CoreMLBaseLayer::BuildLayerType();
}

Status CoreMLBinaryLayer::BuildLayerParam() {
    //layer param
    return CoreMLBaseLayer::BuildLayerParam();
}

Status CoreMLBinaryLayer::BuildConstantWeightsLayer() {
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(layer_resource_);
    if (layer_res && layer_res->element_handle.GetDataCount() > 0) {
        //weight in layer resource
        auto blob_name = GetLayerName();
        blob_name += "-weight-input";
        auto weight_layer = std::make_shared<CoreMLConstLayer>(LAYER_CONST);
        auto status = weight_layer->Init(blob_name, layer_res->element_handle);
        RETURN_ON_NEQ(status, TNN_OK);
        
        coreml_layer_constant_weights_ = {weight_layer};
    } else if(layer_info_ && net_resource_) {
        //weight in constantmap
        return CoreMLBaseLayer::BuildConstantWeightsLayer();
    }
    return TNN_OK;
}

std::vector<std::string> CoreMLBinaryLayer::BuildLayerInputs() {
    if (!layer_info_) {
        return std::vector<std::string>();
    }
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(layer_info_->param.get());
    auto inputs = layer_info_->inputs;
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(layer_resource_);
    if (layer_res && layer_res->element_handle.GetDataCount() > 0) {
        auto weight_name = coreml_layer_constant_weights_[0]->GetLayerName();
        
        auto weight_input_index = layer_param->weight_input_index;
        if (weight_input_index == 0) {
            inputs.insert(inputs.begin(), weight_name);
        } else {
            inputs.push_back(weight_name);
        }
    }
    return inputs;
}

std::vector<std::string> CoreMLBinaryLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

}  // namespace TNN_NS
