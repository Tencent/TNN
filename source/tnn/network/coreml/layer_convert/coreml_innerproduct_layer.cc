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

DECLARE_COREML_LAYER_WITH_DATA(Innerproduct, LAYER_INNER_PRODUCT,
                                std::shared_ptr<CoreML__Specification__WeightParams> bias_param_;
                                std::shared_ptr<CoreML__Specification__WeightParams> weight_param_;);

Status CoreMLInnerproductLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_INNER_PRODUCT;
    return TNN_OK;
}

Status CoreMLInnerproductLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<InnerProductLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto num_output = layer_param->num_output;
    auto has_bias = layer_param->has_bias;
    auto transpose = layer_param->transpose;
    auto axis = layer_param->axis;
    
    auto layer_res = dynamic_cast<InnerProductLayerResource *>(resource_);
    CHECK_PARAM_NULL(layer_res);
    auto weight_size = layer_res->weight_handle.GetDataCount();
    auto weight_ptr = layer_res->weight_handle.force_to<float *>();
    auto weight_dims = layer_res->weight_handle.GetBufferDims();
    auto bias_size = layer_res->bias_handle.GetDataCount();
    auto bias_ptr = layer_res->bias_handle.force_to<float *>();
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__InnerProductLayerParams>(new CoreML__Specification__InnerProductLayerParams);
    coreml_layer_->innerproduct = (CoreML__Specification__InnerProductLayerParams *)coreml_layer_param_.get();
    core_ml__specification__inner_product_layer_params__init(coreml_layer_->innerproduct);
    coreml_layer_->innerproduct->inputchannels = weight_dims.back();
    coreml_layer_->innerproduct->outputchannels = num_output;
    
    weight_param_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
    coreml_layer_->innerproduct->weights = weight_param_.get();
        core_ml__specification__weight_params__init(coreml_layer_->innerproduct->weights);
    coreml_layer_->innerproduct->weights->n_floatvalue = weight_size;
    coreml_layer_->innerproduct->weights->floatvalue = weight_ptr;
    if(bias_ptr) {
        coreml_layer_->innerproduct->hasbias = true;
        bias_param_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
        coreml_layer_->innerproduct->bias = bias_param_.get();
        core_ml__specification__weight_params__init(coreml_layer_->innerproduct->bias);
        coreml_layer_->innerproduct->bias->n_floatvalue = bias_size;
        coreml_layer_->innerproduct->bias->floatvalue = bias_ptr;
    }
    
    return TNN_OK;
}

Status CoreMLInnerproductLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLInnerproductLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLInnerproductLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Innerproduct, LAYER_INNER_PRODUCT);

}  // namespace TNN_NS
