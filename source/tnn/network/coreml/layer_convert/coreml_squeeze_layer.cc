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

DECLARE_COREML_LAYER_WITH_DATA(Squeeze, LAYER_SQUEEZE,
                               std::shared_ptr<void> coreml_layer_axes_;);

Status CoreMLSqueezeLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SQUEEZE;
    return TNN_OK;
}

Status CoreMLSqueezeLayer::BuildLayerParam() {
    //layer param
     auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<SqueezeLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto axes = layer_param->axes;
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__SqueezeLayerParams>(new CoreML__Specification__SqueezeLayerParams);
    coreml_layer_->squeeze = (CoreML__Specification__SqueezeLayerParams *)coreml_layer_param_.get();
    core_ml__specification__squeeze_layer_params__init(coreml_layer_->squeeze);
    if(axes.size() == 0){
        coreml_layer_->squeeze->squeezeall = true;
        coreml_layer_->squeeze->n_axes = axes.size();
    } else {
        coreml_layer_->squeeze->n_axes = axes.size();
        coreml_layer_axes_ = std::shared_ptr<int64_t>(new int64_t [axes.size()], [](int64_t* p) { delete[] p; });
        coreml_layer_->squeeze->axes = (int64_t*) coreml_layer_axes_.get();
        for(int i=0; i<axes.size(); i++){
            coreml_layer_->squeeze->axes[i] = axes[i];
        }
    }
    
    return TNN_OK;
}

Status CoreMLSqueezeLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLSqueezeLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLSqueezeLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Squeeze, LAYER_SQUEEZE);

}  // namespace TNN_NS
