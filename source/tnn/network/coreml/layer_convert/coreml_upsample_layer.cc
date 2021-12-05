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

DECLARE_COREML_LAYER_WITH_DATA(Upsample, LAYER_UPSAMPLE,
                               std::shared_ptr<void> fractionalscalingfactor_;
                               std::shared_ptr<void> scalingfactor_;);

Status CoreMLUpsampleLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_UPSAMPLE;
    return TNN_OK;
}

Status CoreMLUpsampleLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<UpsampleLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto mode = layer_param->mode;
    auto align_corners = layer_param->align_corners;
    auto scales = layer_param->scales;
    auto dims = layer_param->dims;
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__UpsampleLayerParams>(new CoreML__Specification__UpsampleLayerParams);
    coreml_layer_->upsample = (CoreML__Specification__UpsampleLayerParams *)coreml_layer_param_.get();
    core_ml__specification__upsample_layer_params__init(coreml_layer_->upsample);
    coreml_layer_->upsample->n_fractionalscalingfactor = scales.size();
    // Only one of scalingFactor and fractionalScalingFactor can be set, and if set, must be of size 2.
    // scales = fractionalscalingfactor
    fractionalscalingfactor_ = std::shared_ptr<float>(new float [scales.size()], [](float* p) { delete[] p; });
    coreml_layer_->upsample->fractionalscalingfactor = (float *) fractionalscalingfactor_.get();
    for(int i=0; i<scales.size(); i++ ) {
        coreml_layer_->upsample->fractionalscalingfactor[i] = scales[i];
    }
    
    if(mode == 1) {  // nearest
        coreml_layer_->upsample->mode = CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__INTERPOLATION_MODE__NN;
    } else if(mode == 2) {  // bilinear/linear
        coreml_layer_->upsample->mode = CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__INTERPOLATION_MODE__BILINEAR;
        // align corners option from pytorch
        if(align_corners == 0) {  // ALIGN_CORNERS_TRUE: spacing = (Xin-1) / (Xout-1) grid_point[i] = min(Xin-1, max(0, i * spacing)), for i = 0,1,2,….,Xout-1
            coreml_layer_->upsample->linearupsamplemode=   CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__LINEAR_UPSAMPLE_MODE__ALIGN_CORNERS_TRUE;
        } else if(align_corners == 1) {  // ALIGN_CORNERS_FALSE: spacing = Xin / Xout grid_point[i] = min(Xin-1, max(0, i * spacing + 0.5 * spacing - 0.5)), for i = 0,1,2,….,Xout-1
            coreml_layer_->upsample->linearupsamplemode= CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__LINEAR_UPSAMPLE_MODE__ALIGN_CORNERS_FALSE;
        }
    } else { // cubic ...
        LOGE("Error: Upsample dont support resize type\n");
        return Status(TNNERR_MODEL_ERR, "Error: Upsample dont support resize type");
    }
    
    return TNN_OK;
}

Status CoreMLUpsampleLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLUpsampleLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLUpsampleLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Upsample, LAYER_UPSAMPLE);

}  // namespace TNN_NS
