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

DECLARE_COREML_LAYER(Softmax, LAYER_SOFTMAX);

Status CoreMLSoftmaxLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SOFTMAX_ND;
    return TNN_OK;
}

Status CoreMLSoftmaxLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<SoftmaxLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto axis = layer_param->axis;
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__SoftmaxNDLayerParams>(new CoreML__Specification__SoftmaxNDLayerParams);
    coreml_layer_->softmaxnd = (CoreML__Specification__SoftmaxNDLayerParams *)coreml_layer_param_.get();
    core_ml__specification__softmax_ndlayer_params__init(coreml_layer_->softmaxnd);
    coreml_layer_->softmaxnd->axis = axis;
    
    return TNN_OK;
}

Status CoreMLSoftmaxLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLSoftmaxLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLSoftmaxLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Softmax, LAYER_SOFTMAX);

}  // namespace TNN_NS
