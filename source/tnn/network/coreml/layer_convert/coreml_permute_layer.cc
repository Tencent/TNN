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

DECLARE_COREML_LAYER_WITH_DATA(Permute, LAYER_PERMUTE,
                                std::shared_ptr<void> coreml_layer_axes_;);

Status CoreMLPermuteLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_TRANSPOSE;
    return TNN_OK;
}

Status CoreMLPermuteLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<PermuteLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto orders = layer_param->orders;
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__TransposeLayerParams>(new CoreML__Specification__TransposeLayerParams);
    coreml_layer_->transpose = (CoreML__Specification__TransposeLayerParams *)coreml_layer_param_.get();
    core_ml__specification__transpose_layer_params__init(coreml_layer_->transpose);
    coreml_layer_->transpose->n_axes = orders.size();
    coreml_layer_axes_ = std::shared_ptr<uint64_t>(new uint64_t [coreml_layer_->transpose->n_axes], [](uint64_t* p) { delete[] p; });
    coreml_layer_->transpose->axes = (uint64_t *)coreml_layer_axes_.get();
    for(int i=0;i<orders.size();i++){
        coreml_layer_->transpose->axes[i] = orders[i];
    }
    
    return TNN_OK;
}

Status CoreMLPermuteLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLPermuteLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLPermuteLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Permute, LAYER_PERMUTE);
}  // namespace TNN_NS
