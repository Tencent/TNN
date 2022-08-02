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

DECLARE_COREML_LAYER(Gather, LAYER_GATHER);

Status CoreMLGatherLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_GATHER;
    return TNN_OK;
}

Status CoreMLGatherLayer::BuildLayerParam() {
    GatherLayerParam *layer_param = layer_info_ ? dynamic_cast<GatherLayerParam *>(layer_info_->param.get()) : nullptr;
    CHECK_PARAM_NULL(layer_param);
    
    //layer param
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__GatherLayerParams>(new CoreML__Specification__GatherLayerParams);
    coreml_layer_->gather = (CoreML__Specification__GatherLayerParams *)coreml_layer_param_.get();
    core_ml__specification__gather_layer_params__init(coreml_layer_->gather);
    coreml_layer_->gather->axis = layer_param->axis;
    return TNN_OK;
}

Status CoreMLGatherLayer::BuildConstantWeightsLayer() {
    RETURN_ON_NEQ(CoreMLBaseLayer::BuildConstantWeightsLayer(), TNN_OK);
    
    GatherLayerParam *layer_param = layer_info_ ? dynamic_cast<GatherLayerParam *>(layer_info_->param.get()) : nullptr;
    CHECK_PARAM_NULL(layer_param);
    if (!layer_param->data_in_resource && !layer_param->indices_in_resource) {
        return TNN_OK;
    }
    
    auto layer_res = dynamic_cast<GatherLayerResource *>(layer_resource_);
    if (!layer_res) {
        return Status(TNNERR_MODEL_ERR, "Gather resource is invalid");
    }
    
    coreml_layer_constant_weights_.clear();
    if (layer_param->data_in_resource && layer_res->data.GetDataCount() > 0) {
        auto blob_name = GetLayerName();
        blob_name += "-input-data";
        auto weight = std::make_shared<CoreMLConstLayer>(LAYER_CONST);
        auto status = weight->Init(blob_name, layer_res->data);
        RETURN_ON_NEQ(status, TNN_OK);
        coreml_layer_constant_weights_ = {weight};
    }
    
    
    if (layer_param->indices_in_resource && layer_res->indices.GetDataCount() > 0) {
        auto blob_name = GetLayerName();
        blob_name += "-input-indices";
        auto weight = std::make_shared<CoreMLConstLayer>(LAYER_CONST);
        auto status = weight->Init(blob_name, layer_res->indices);
        RETURN_ON_NEQ(status, TNN_OK);
        coreml_layer_constant_weights_ .push_back(weight);
    }
    return TNN_OK;
}

std::vector<std::string> CoreMLGatherLayer::BuildLayerInputs() {
    GatherLayerParam *layer_param = layer_info_ ? dynamic_cast<GatherLayerParam *>(layer_info_->param.get()) : nullptr;
    if (!layer_param || layer_info_->inputs.size() < 1) {
        LOGE("CoreMLGatherLayer has invalid layer_info or layer_param or inputs size\n");
        return std::vector<std::string>();
    }
    
    std::string data_name="";
    std::string indices_name="";
    if (layer_param->data_in_resource) {
        data_name = GetLayerName() + "-input-data";
    } else {
        data_name = layer_info_->inputs.front();
    }
    
    if (layer_param->indices_in_resource) {
        indices_name = GetLayerName() + "-input-indices";
    } else {
        indices_name = layer_info_->inputs.back();
    }
    return {data_name, indices_name};
}

std::vector<std::string> CoreMLGatherLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Gather, LAYER_GATHER);

}  // namespace TNN_NS
