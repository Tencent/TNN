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

// Use Activation Tanh
namespace TNN_NS {

DECLARE_COREML_LAYER_WITH_DATA(Gelu, LAYER_GELU,
                                std::shared_ptr<void> coreml_layer_type_;);

Status CoreMLGeluLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_GELU;
    return TNN_OK;
}

Status CoreMLGeluLayer::BuildLayerParam() {
    //layer param
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__GeluLayerParams>(new CoreML__Specification__GeluLayerParams);
    coreml_layer_->gelu = (CoreML__Specification__GeluLayerParams *)coreml_layer_param_.get();
    core_ml__specification__gelu_layer_params__init(coreml_layer_->gelu);
    
    coreml_layer_->gelu->mode = CORE_ML__SPECIFICATION__GELU_LAYER_PARAMS__GELU_MODE__EXACT;
    
    return TNN_OK;
}

Status CoreMLGeluLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLGeluLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLGeluLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Gelu, LAYER_GELU);

}  // namespace TNN_NS
