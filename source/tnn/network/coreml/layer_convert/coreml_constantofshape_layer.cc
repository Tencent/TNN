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

DECLARE_COREML_LAYER(ConstantOfShape, LAYER_CONSTANT_OF_SHAPE);

Status CoreMLConstantOfShapeLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_FILL_DYNAMIC;
    return TNN_OK;
}

Status CoreMLConstantOfShapeLayer::BuildLayerParam() {
    //layer param
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__FillDynamicLayerParams>(new CoreML__Specification__FillDynamicLayerParams);
    coreml_layer_->filldynamic = (CoreML__Specification__FillDynamicLayerParams *)coreml_layer_param_.get();
    core_ml__specification__fill_dynamic_layer_params__init(coreml_layer_->filldynamic);
    
    auto layer_resource = dynamic_cast<ConstantOfShapeLayerResource*>(layer_resource_);
    CHECK_PARAM_NULL(layer_resource);
    
    auto data_count = layer_resource->value.GetDataCount();
    if (data_count <= 0) {
        LOGE("CoreMLConstantOfShapeLayer has invalide data count\n");
        return Status(TNNERR_MODEL_ERR, "CoreMLConstantOfShapeLayer has invalide data count");
    }
    auto data_type = layer_resource->value.GetDataType();
    if (data_type == DATA_TYPE_FLOAT) {
        coreml_layer_->filldynamic->value = *(layer_resource->value.force_to<float *>());
    } else {
        LOGE("CoreMLConstantOfShapeLayer dont support data type %d\n", data_type);
        return Status(TNNERR_MODEL_ERR, "CoreMLConstantOfShapeLayer dont support data type");
    }
    return TNN_OK;
}

Status CoreMLConstantOfShapeLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLConstantOfShapeLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLConstantOfShapeLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(ConstantOfShape, LAYER_CONSTANT_OF_SHAPE);

}  // namespace TNN_NS
