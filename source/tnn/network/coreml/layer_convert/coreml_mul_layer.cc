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
#include "coreml_const_layer.h"

namespace TNN_NS {

DECLARE_COREML_LAYER(Mul, LAYER_MUL);

Status CoreMLMulLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_MULTIPLY_BROADCASTABLE;
    return TNN_OK;
}

Status CoreMLMulLayer::BuildLayerParam() {
    //layer param
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__MultiplyBroadcastableLayerParams>(new CoreML__Specification__MultiplyBroadcastableLayerParams);
    coreml_layer_->multiplybroadcastable = (CoreML__Specification__MultiplyBroadcastableLayerParams *)coreml_layer_param_.get();
    core_ml__specification__multiply_broadcastable_layer_params__init(coreml_layer_->multiplybroadcastable);
    return TNN_OK;
}

Status CoreMLMulLayer::BuildConstantWeightsLayer() {
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(layer_resource_);
    if (layer_res && layer_res->element_handle.GetDataCount() > 0) {
        auto blob_name = GetLayerName();
        blob_name += "-weight-input";
        auto weight = std::make_shared<CoreMLConstLayer>(LAYER_CONST);
        auto status = weight->Init(blob_name, layer_res->element_handle);
        RETURN_ON_NEQ(status, TNN_OK);
        
        coreml_layer_constant_weights_ = {weight};
    }
    return TNN_OK;
}

std::vector<std::string> CoreMLMulLayer::BuildLayerInputs() {
    if (!layer_info_) {
        return std::vector<std::string>();
    }
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(layer_info_->param.get());
    auto inputs = layer_info_->inputs;
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(layer_resource_);
    if (layer_res && coreml_layer_constant_weights_.size() > 0) {
        auto weight_name = coreml_layer_constant_weights_[0]->GetLayerName();
        
        auto weight_input_index = layer_param->weight_input_index;
        if (weight_input_index == 0) {
            inputs.insert(inputs.begin(), weight_name);
        } else {
            inputs.push_back(weight_name);
        }
    }
    return inputs;
}

std::vector<std::string> CoreMLMulLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Mul, LAYER_MUL);

}  // namespace TNN_NS
