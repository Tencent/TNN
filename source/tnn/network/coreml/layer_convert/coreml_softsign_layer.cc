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

DECLARE_COREML_LAYER_WITH_DATA(Softsign, LAYER_SOFTSIGN,
                                std::shared_ptr<void> coreml_layer_type_;);

Status CoreMLSoftsignLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ACTIVATION;
    return TNN_OK;
}

Status CoreMLSoftsignLayer::BuildLayerParam() {
    //layer param
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__ActivationParams>(new CoreML__Specification__ActivationParams);
    coreml_layer_->activation = (CoreML__Specification__ActivationParams *)coreml_layer_param_.get();
    core_ml__specification__activation_params__init(coreml_layer_->activation);
    coreml_layer_->activation->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_SOFTSIGN;
    coreml_layer_type_ = std::shared_ptr<CoreML__Specification__ActivationSoftsign>(new CoreML__Specification__ActivationSoftsign);
    coreml_layer_->activation->softsign = (CoreML__Specification__ActivationSoftsign *)coreml_layer_type_.get();
    core_ml__specification__activation_softsign__init(coreml_layer_->activation->softsign);
    
    return TNN_OK;
}

Status CoreMLSoftsignLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLSoftsignLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLSoftsignLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Softsign, LAYER_SOFTSIGN);

}  // namespace TNN_NS
