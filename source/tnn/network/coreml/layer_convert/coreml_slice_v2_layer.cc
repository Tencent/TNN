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

DECLARE_COREML_LAYER_WITH_DATA(SliceV2, LAYER_STRIDED_SLICE_V2,
                                std::shared_ptr<void> coreml_layer_type_;);

Status CoreMLSliceV2Layer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SLICE;
    return TNN_OK;
}

Status CoreMLSliceV2Layer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<StrideSliceV2LayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto begins = layer_param->begins;
    auto ends = layer_param->ends;
    auto axes = layer_param->axes;
    auto strides = layer_param->strides;
    
    std::vector<int> input_shape;
    if (net_resource_ && layer_info_->inputs.size()>0 && layer_info_->outputs.size()>0) {
        if (net_resource_->blob_shapes_map.find(layer_info_->inputs[0]) != net_resource_->blob_shapes_map.end()) {
            input_shape = net_resource_->blob_shapes_map[layer_info_->inputs[0]];
        }
    }
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__SliceLayerParams>(new CoreML__Specification__SliceLayerParams);
    coreml_layer_->slice = (CoreML__Specification__SliceLayerParams *)coreml_layer_param_.get();
    core_ml__specification__slice_layer_params__init(coreml_layer_->slice);
    switch (axes.front()) {
        case 1:
            coreml_layer_->slice->axis = CORE_ML__SPECIFICATION__SLICE_LAYER_PARAMS__SLICE_AXIS__CHANNEL_AXIS;
            break;
        case 2:
            coreml_layer_->slice->axis = CORE_ML__SPECIFICATION__SLICE_LAYER_PARAMS__SLICE_AXIS__HEIGHT_AXIS;
            break;
        case 3:
            coreml_layer_->slice->axis = CORE_ML__SPECIFICATION__SLICE_LAYER_PARAMS__SLICE_AXIS__WIDTH_AXIS;
            break;
        default:
            LOGE("Error: SliceLayer failed, dont support axes:%d\n", axes.front());
            return Status(TNNERR_PARAM_ERR, "SliceLayer failed, dont support this axes");
            break;
    }
    coreml_layer_->slice->startindex = begins.front();
    coreml_layer_->slice->endindex = ends.front() > input_shape[axes.front()] ? input_shape[axes.front()] : ends.front();
    coreml_layer_->slice->stride = strides.front();
    
    return TNN_OK;
}

Status CoreMLSliceV2Layer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLSliceV2Layer::BuildLayerInputs() {
    return {layer_info_->inputs[0]};
}

std::vector<std::string> CoreMLSliceV2Layer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(SliceV2, LAYER_STRIDED_SLICE_V2);

}  // namespace TNN_NS
