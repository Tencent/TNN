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

DECLARE_COREML_BINARY_LAYER(Sub, LAYER_SUB);

Status CoreMLSubLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SUBTRACT_BROADCASTABLE;
    return TNN_OK;
}

Status CoreMLSubLayer::BuildLayerParam() {
    //layer param
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__SubtractBroadcastableLayerParams>(new CoreML__Specification__SubtractBroadcastableLayerParams);
    coreml_layer_->subtractbroadcastable = (CoreML__Specification__SubtractBroadcastableLayerParams *)coreml_layer_param_.get();
    core_ml__specification__subtract_broadcastable_layer_params__init(coreml_layer_->subtractbroadcastable);
    return TNN_OK;
}

Status CoreMLSubLayer::BuildConstantWeightsLayer() {
    return CoreMLBinaryLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLSubLayer::BuildLayerInputs() {
    return CoreMLBinaryLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLSubLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Sub, LAYER_SUB);

}  // namespace TNN_NS
