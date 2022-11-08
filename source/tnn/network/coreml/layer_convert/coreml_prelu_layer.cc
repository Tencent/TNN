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

DECLARE_COREML_LAYER_WITH_DATA(PRelu, LAYER_PRELU,
                                std::shared_ptr<void> coreml_layer_type_;
                                std::shared_ptr<CoreML__Specification__WeightParams> coreml_layer_alpha_;
                                std::shared_ptr<RawBuffer> rawbuffer_fp32_slope_;);

Status CoreMLPReluLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_ACTIVATION;
    return TNN_OK;
}

Status CoreMLPReluLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<PReluLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto channel_share = layer_param->channel_shared;
    auto has_filter = layer_param->has_filler;
    
    auto layer_res = dynamic_cast<PReluLayerResource *>(layer_resource_);
    CHECK_PARAM_NULL(layer_res);
    auto slope_count = layer_res->slope_handle.GetDataCount();
    auto slope_data_type = layer_res->slope_handle.GetDataType();
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__ActivationParams>(new CoreML__Specification__ActivationParams);
    coreml_layer_->activation = (CoreML__Specification__ActivationParams *)coreml_layer_param_.get();
    core_ml__specification__activation_params__init(coreml_layer_->activation);
    
    if (channel_share) {  // Leaky ReLU
        coreml_layer_->activation->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_LEAKY_RE_LU;
        coreml_layer_type_ = std::shared_ptr<CoreML__Specification__ActivationLeakyReLU>(new CoreML__Specification__ActivationLeakyReLU);
        coreml_layer_->activation->leakyrelu = (CoreML__Specification__ActivationLeakyReLU *)coreml_layer_type_.get();
        core_ml__specification__activation_leaky_re_lu__init(coreml_layer_->activation->leakyrelu);
        coreml_layer_->activation->leakyrelu->alpha = *(layer_res->slope_handle.force_to<float *>());
    } else {  // PReLU
        coreml_layer_->activation->nonlinearity_type_case = CORE_ML__SPECIFICATION__ACTIVATION_PARAMS__NONLINEARITY_TYPE_PRE_LU;
        coreml_layer_type_ = std::shared_ptr<CoreML__Specification__ActivationPReLU>(new CoreML__Specification__ActivationPReLU);
        coreml_layer_->activation->prelu = (CoreML__Specification__ActivationPReLU *)coreml_layer_type_.get();
        core_ml__specification__activation_pre_lu__init(coreml_layer_->activation->prelu);
        
        RETURN_ON_NEQ(RawBuffer2CoreMLWeight(&(layer_res->slope_handle),
                                             coreml_layer_alpha_, rawbuffer_fp32_slope_), TNN_OK);
        coreml_layer_->activation->prelu->alpha =  coreml_layer_alpha_.get();
    }
    
    return TNN_OK;
}

Status CoreMLPReluLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLPReluLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLPReluLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(PRelu, LAYER_PRELU);

}  // namespace TNN_NS
