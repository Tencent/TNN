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

DECLARE_COREML_LAYER_WITH_DATA(Clip, LAYER_CLIP,
                                std::shared_ptr<void> coreml_layer_type_;);

Status CoreMLClipLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_CLIP;
    return TNN_OK;
}

Status CoreMLClipLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<ClipLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto min = layer_param->min;
    auto max = layer_param->max;
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__ClipLayerParams>(new CoreML__Specification__ClipLayerParams);
    coreml_layer_->clip = (CoreML__Specification__ClipLayerParams *)coreml_layer_param_.get();
    core_ml__specification__clip_layer_params__init(coreml_layer_->clip);
    coreml_layer_->clip->minval = min;
    coreml_layer_->clip->maxval = max;
    
    return TNN_OK;
}

Status CoreMLClipLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLClipLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLClipLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Clip, LAYER_CLIP);

}  // namespace TNN_NS
