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

DECLARE_COREML_LAYER_WITH_DATA(ReduceSum, LAYER_REDUCE_SUM,
                                std::shared_ptr<void> coreml_layer_axis_;);

Status CoreMLReduceSumLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_REDUCE_SUM;
    return TNN_OK;
}

Status CoreMLReduceSumLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<ReduceLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto keep_dims = layer_param->keep_dims;
    auto axis = layer_param->axis;
    // ignore axis, reduce all to one
    auto all_reduce = layer_param->all_reduce;
   
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__ReduceSumLayerParams>(new CoreML__Specification__ReduceSumLayerParams);
    coreml_layer_->reducesum = (CoreML__Specification__ReduceSumLayerParams *)coreml_layer_param_.get();
    core_ml__specification__reduce_sum_layer_params__init(coreml_layer_->reducesum);
    coreml_layer_->reducesum->keepdims = keep_dims;
    coreml_layer_->reducesum->reduceall = all_reduce;
    coreml_layer_->reducesum->n_axes = axis.size();
    
    coreml_layer_axis_ = std::shared_ptr<int64_t>(new int64_t [axis.size()], [](int64_t* p) { delete[] p; });
    coreml_layer_->reducesum->axes = (int64_t *)coreml_layer_axis_.get();
    for(int i = 0; i < axis.size(); i++){
        coreml_layer_->reducesum->axes[i] = axis[i];
    }

    return TNN_OK;
}

Status CoreMLReduceSumLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLReduceSumLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLReduceSumLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(ReduceSum, LAYER_REDUCE_SUM);

}  // namespace TNN_NS
