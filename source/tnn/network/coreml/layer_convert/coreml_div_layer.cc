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

DECLARE_COREML_BINARY_LAYER(Div, LAYER_DIV);

Status CoreMLDivLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_DIVIDE_BROADCASTABLE;
    return TNN_OK;
}

Status CoreMLDivLayer::BuildLayerParam() {
    //layer param
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__DivideBroadcastableLayerParams>(new CoreML__Specification__DivideBroadcastableLayerParams);
    coreml_layer_->dividebroadcastable = (CoreML__Specification__DivideBroadcastableLayerParams *)coreml_layer_param_.get();
    core_ml__specification__divide_broadcastable_layer_params__init(coreml_layer_->dividebroadcastable);
    return TNN_OK;
}

Status CoreMLDivLayer::BuildConstantWeightsLayer() {
    return CoreMLBinaryLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLDivLayer::BuildLayerInputs() {
    return CoreMLBinaryLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLDivLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Div, LAYER_DIV);

}  // namespace TNN_NS
