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

namespace TNN_NS {

DECLARE_COREML_LAYER_WITH_DATA(Exp, LAYER_EXP,
                                std::shared_ptr<void> coreml_layer_type_;);

Status CoreMLExpLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_UNARY;
    return TNN_OK;
}

Status CoreMLExpLayer::BuildLayerParam() {
    //layer param
    coreml_layer_param_ = std::shared_ptr<_CoreML__Specification__UnaryFunctionLayerParams>(new CoreML__Specification__UnaryFunctionLayerParams);
    coreml_layer_->unary = (_CoreML__Specification__UnaryFunctionLayerParams *)coreml_layer_param_.get();
    core_ml__specification__unary_function_layer_params__init(coreml_layer_->unary);
    coreml_layer_->unary->type = CORE_ML__SPECIFICATION__UNARY_FUNCTION_LAYER_PARAMS__OPERATION__EXP;
    
    return TNN_OK;
}

Status CoreMLExpLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLExpLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLExpLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Exp, LAYER_EXP);

}  // namespace TNN_NS
