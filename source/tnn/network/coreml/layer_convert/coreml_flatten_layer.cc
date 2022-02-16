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

DECLARE_COREML_LAYER_WITH_DATA(Flatten, LAYER_FLATTEN,
                                std::shared_ptr<void> coreml_layer_aixs_;);

Status CoreMLFlattenLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_FLATTEN_TO2_D;
    return TNN_OK;
}

Status CoreMLFlattenLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<FlattenLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto aixs = layer_param->axis;
    
    int input_shape_size,output_shape_size;
    if (net_resource_ && layer_info_->inputs.size()>0 && layer_info_->outputs.size()>0) {
        if (net_resource_->blob_shapes_map.find(layer_info_->inputs[0]) != net_resource_->blob_shapes_map.end()) {
            auto input_shape = net_resource_->blob_shapes_map[layer_info_->inputs[0]];
            input_shape_size = (int)input_shape.size();
        }
        
        if (net_resource_->blob_shapes_map.find(layer_info_->outputs[0]) != net_resource_->blob_shapes_map.end()) {
            auto output_shape = net_resource_->blob_shapes_map[layer_info_->outputs[0]];
            output_shape_size = (int)output_shape.size();
        }
    }
    
    if(output_shape_size != 2){
        return Status(TNNERR_MODEL_ERR, "CoreMLFlattenLayer output ranks must be 2 dims");
    }
    auto aixs_ = aixs - input_shape_size;
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__FlattenTo2DLayerParams>(new CoreML__Specification__FlattenTo2DLayerParams);
    coreml_layer_->flattento2d = (CoreML__Specification__FlattenTo2DLayerParams *)coreml_layer_param_.get();
    core_ml__specification__flatten_to2_dlayer_params__init(coreml_layer_->flattento2d);
    coreml_layer_->flattento2d->axis = aixs_;
    
    return TNN_OK;
}

Status CoreMLFlattenLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLFlattenLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLFlattenLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Flatten, LAYER_FLATTEN);
}  // namespace TNN_NS
