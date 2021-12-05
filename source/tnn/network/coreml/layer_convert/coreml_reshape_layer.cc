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

DECLARE_COREML_LAYER_WITH_DATA(Reshape, LAYER_RESHAPE,
                                std::shared_ptr<void> coreml_layer_shape_;
                                std::shared_ptr<CoreML__Specification__Tensor*> coreml_layer_inputtensor_arr_;
                                std::vector<std::shared_ptr<CoreML__Specification__Tensor> > coreml_layer_inputtensor_;
                                std::shared_ptr<CoreML__Specification__Tensor*> coreml_layer_outputtensor_arr_;
                                std::vector<std::shared_ptr<CoreML__Specification__Tensor> > coreml_layer_outputtensor_;);

Status CoreMLReshapeLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_RANK_PRESERVING_RESHAPE;
    return TNN_OK;
}

Status CoreMLReshapeLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
    CHECK_PARAM_NULL(reshape_param);
    auto input_size = layer_info_->inputs.size();
    auto output_size = layer_info_->outputs.size();
    auto shape_size = reshape_param->shape.size();
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__RankPreservingReshapeLayerParams>(new CoreML__Specification__RankPreservingReshapeLayerParams);
    coreml_layer_->rankpreservingreshape = (CoreML__Specification__RankPreservingReshapeLayerParams *)coreml_layer_param_.get();
    core_ml__specification__rank_preserving_reshape_layer_params__init(coreml_layer_->rankpreservingreshape);
    coreml_layer_->rankpreservingreshape->n_targetshape = shape_size;
    coreml_layer_shape_ = std::shared_ptr<int64_t>(new int64_t [coreml_layer_->rankpreservingreshape->n_targetshape], [](int64_t* p) { delete[] p; });
    coreml_layer_->rankpreservingreshape->targetshape = (int64_t *)coreml_layer_shape_.get();
    for(int i = 0; i < shape_size; i++){
        coreml_layer_->rankpreservingreshape->targetshape[i] = reshape_param->shape[i];
    }
    
    //input & output rank must be equal!
    //set inputtensor rank
    coreml_layer_->n_inputtensor = input_size;
    coreml_layer_inputtensor_arr_ = std::shared_ptr<CoreML__Specification__Tensor*>(new CoreML__Specification__Tensor* [input_size], [](CoreML__Specification__Tensor** p) { delete[] p; });
    coreml_layer_->inputtensor = coreml_layer_inputtensor_arr_.get();
    for(int i=0; i<input_size; i++){
        coreml_layer_inputtensor_.push_back(std::shared_ptr<CoreML__Specification__Tensor>(new CoreML__Specification__Tensor));
        coreml_layer_->inputtensor[i] = coreml_layer_inputtensor_[i].get();
        core_ml__specification__tensor__init(coreml_layer_->inputtensor[i]);
        coreml_layer_->inputtensor[i]->rank = (uint32_t)shape_size;
    }
    
    //set outputtensor rank
    coreml_layer_->n_outputtensor = output_size;
    coreml_layer_outputtensor_arr_ = std::shared_ptr<CoreML__Specification__Tensor*>(new CoreML__Specification__Tensor* [output_size], [](CoreML__Specification__Tensor** p) { delete[] p; });
    coreml_layer_->outputtensor = coreml_layer_outputtensor_arr_.get();
    for(int i=0; i<output_size; i++){
        coreml_layer_outputtensor_.push_back(std::shared_ptr<CoreML__Specification__Tensor>(new CoreML__Specification__Tensor));
        coreml_layer_->outputtensor[i] = coreml_layer_outputtensor_[i].get();
        core_ml__specification__tensor__init(coreml_layer_->outputtensor[i]);
        coreml_layer_->outputtensor[i]->rank = (uint32_t)shape_size;
    }
    
    return TNN_OK;
}

Status CoreMLReshapeLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLReshapeLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLReshapeLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Reshape, LAYER_RESHAPE);

}  // namespace TNN_NS
