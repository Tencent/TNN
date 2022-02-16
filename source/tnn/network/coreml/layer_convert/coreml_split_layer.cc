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

DECLARE_COREML_LAYER_WITH_DATA(Split, LAYER_SPLITV,
                                std::shared_ptr<void> coreml_layer_splitsizes_;);

Status CoreMLSplitLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SPLIT_ND;
    return TNN_OK;
}

Status CoreMLSplitLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<SplitVLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto axis = layer_param->axis;
    auto slices = layer_param->slices;
    auto is_split_specified = layer_param->is_split_specified;
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__SplitNDLayerParams>(new CoreML__Specification__SplitNDLayerParams);
    coreml_layer_->splitnd = (CoreML__Specification__SplitNDLayerParams *)coreml_layer_param_.get();
    core_ml__specification__split_ndlayer_params__init(coreml_layer_->splitnd);
    coreml_layer_->splitnd->axis = axis;
    coreml_layer_->splitnd->numsplits = slices.size();
    coreml_layer_->splitnd->n_splitsizes = slices.size();
    coreml_layer_splitsizes_ = std::shared_ptr<uint64_t>(new uint64_t [slices.size()], [](uint64_t* p) { delete[] p; });
    coreml_layer_->splitnd->splitsizes = (uint64_t *)coreml_layer_splitsizes_.get();
    for(int i = 0; i < slices.size(); i++){
        coreml_layer_->splitnd->splitsizes[i] = slices[i];
    }

    return TNN_OK;
}

Status CoreMLSplitLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLSplitLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLSplitLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Split, LAYER_SPLITV);

}  // namespace TNN_NS
