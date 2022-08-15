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

DECLARE_COREML_LAYER(Concat, LAYER_CONCAT);

Status CoreMLConcatLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_CONCAT_ND;
    return TNN_OK;
}

Status CoreMLConcatLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto concat_param = dynamic_cast<ConcatLayerParam *>(param);
    CHECK_PARAM_NULL(concat_param);
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__ConcatNDLayerParams>(new CoreML__Specification__ConcatNDLayerParams);
    coreml_layer_->concatnd = (CoreML__Specification__ConcatNDLayerParams *)coreml_layer_param_.get();
    core_ml__specification__concat_ndlayer_params__init(coreml_layer_->concatnd);
    coreml_layer_->concatnd->axis = concat_param->axis;
    
    return TNN_OK;
}

Status CoreMLConcatLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLConcatLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLConcatLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Concat, LAYER_CONCAT);

}  // namespace TNN_NS
