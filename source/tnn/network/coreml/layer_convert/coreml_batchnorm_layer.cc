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

DECLARE_COREML_LAYER_WITH_DATA(Batchnorm, LAYER_BATCH_NORM,
                                std::shared_ptr<void> coreml_layer_type_;
                                std::shared_ptr<void> coreml_layer_gamma_;
                                std::shared_ptr<void> coreml_layer_beta_;
                                std::shared_ptr<void> coreml_layer_mean_;
                                std::shared_ptr<void> coreml_layer_variance_;
                                std::shared_ptr<void> mean_;
                                std::shared_ptr<void> variance_;);

Status CoreMLBatchnormLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_BATCHNORM;
    return TNN_OK;
}

Status CoreMLBatchnormLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<BatchNormLayerParam *>(param);
//    CHECK_PARAM_NULL(layer_param);
//    auto channels = layer_param->channels;
    
    auto layer_res = dynamic_cast<BatchNormLayerResource *>(resource_);
    CHECK_PARAM_NULL(layer_res);
    auto scale_count = layer_res->scale_handle.GetDataCount();
    auto scale_ptr = layer_res->scale_handle.force_to<float *>();
    auto bias_count = layer_res->bias_handle.GetDataCount();
    auto bias_ptr = layer_res->bias_handle.force_to<float *>();
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__BatchnormLayerParams>(new CoreML__Specification__BatchnormLayerParams);
    coreml_layer_->batchnorm = (CoreML__Specification__BatchnormLayerParams *)coreml_layer_param_.get();
    core_ml__specification__batchnorm_layer_params__init(coreml_layer_->batchnorm);
    coreml_layer_->batchnorm->channels = scale_count;
    coreml_layer_->batchnorm->computemeanvar = false;
    coreml_layer_->batchnorm->instancenormalization = false;
    coreml_layer_gamma_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
    coreml_layer_->batchnorm->gamma = (CoreML__Specification__WeightParams*) coreml_layer_gamma_.get();
    core_ml__specification__weight_params__init(coreml_layer_->batchnorm->gamma);
    coreml_layer_->batchnorm->gamma->n_floatvalue = scale_count;
    coreml_layer_->batchnorm->gamma->floatvalue = scale_ptr;
    coreml_layer_beta_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
    coreml_layer_->batchnorm->beta = (CoreML__Specification__WeightParams*) coreml_layer_beta_.get();
    core_ml__specification__weight_params__init(coreml_layer_->batchnorm->beta);
    coreml_layer_->batchnorm->beta->n_floatvalue = bias_count;
    coreml_layer_->batchnorm->beta->floatvalue = bias_ptr;
    coreml_layer_mean_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
    coreml_layer_->batchnorm->mean = (CoreML__Specification__WeightParams*) coreml_layer_mean_.get();
    core_ml__specification__weight_params__init(coreml_layer_->batchnorm->mean);
    coreml_layer_->batchnorm->mean->n_floatvalue = scale_count;
    mean_ = std::shared_ptr<float>(new float[scale_count], [](float* p) { delete[] p; });
    coreml_layer_->batchnorm->mean->floatvalue = (float*) mean_.get();
    for(int i=0; i<scale_count; i++){
        coreml_layer_->batchnorm->mean->floatvalue[i] = -1;
    }
    coreml_layer_variance_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
    coreml_layer_->batchnorm->variance = (CoreML__Specification__WeightParams*) coreml_layer_variance_.get();
    core_ml__specification__weight_params__init(coreml_layer_->batchnorm->variance);
    coreml_layer_->batchnorm->variance->n_floatvalue = scale_count;
    variance_ = std::shared_ptr<float>(new float[scale_count], [](float* p) { delete[] p; });
    coreml_layer_->batchnorm->variance->floatvalue = (float*) variance_.get();
    for(int i=0; i<scale_count; i++){
        coreml_layer_->batchnorm->variance->floatvalue[i] = 1;
    }
    
    return TNN_OK;
}

Status CoreMLBatchnormLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLBatchnormLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLBatchnormLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Batchnorm, LAYER_BATCH_NORM);

}  // namespace TNN_NS
