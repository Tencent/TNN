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
    auto scales = layer_param->scales;    // order [w h d]
    auto dims = layer_param->dims;    // order [w h d]
    
    std::vector<int> input_shape;
    int input_shape_size = 0;
    if (net_resource_ && layer_info_->inputs.size()>0 && layer_info_->outputs.size()>0) {
        if (net_resource_->blob_shapes_map.find(layer_info_->inputs[0]) != net_resource_->blob_shapes_map.end()) {
            input_shape = net_resource_->blob_shapes_map[layer_info_->inputs[0]];
            input_shape_size = input_shape.size();
        }
    }
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__UpsampleLayerParams>(new CoreML__Specification__UpsampleLayerParams);
    coreml_layer_->upsample = (CoreML__Specification__UpsampleLayerParams *)coreml_layer_param_.get();
    core_ml__specification__upsample_layer_params__init(coreml_layer_->upsample);
    // Only one of scalingFactor and fractionalScalingFactor can be set, and if set, must be of size 2.
    // scales = fractionalscalingfactor
    bool isFractional = false;
    if (dims.size() > 0) {
        std::vector<float> scales_ = {};
        for(int i=0; i<dims.size(); i++){
            scales_.push_back(float(dims[dims.size() - i - 1]) / input_shape[input_shape_size - dims.size() + i]);
        }
        for(int i=0;i<scales_.size();i++){
            if((scales_[i]-((int)scales_[i])) > 0.000001){
                isFractional = true;
            }
        }
        if(isFractional){
            coreml_layer_->upsample->n_fractionalscalingfactor = dims.size();
            fractionalscalingfactor_ = std::shared_ptr<float>(new float [dims.size()], [](float* p) { delete[] p; });
            coreml_layer_->upsample->fractionalscalingfactor = (float *) fractionalscalingfactor_.get();
            for(int i=0; i<dims.size(); i++){
                coreml_layer_->upsample->fractionalscalingfactor[i] = scales_[i];
            }
        } else {
            coreml_layer_->upsample->n_scalingfactor = dims.size();
            scalingfactor_ = std::shared_ptr<uint64_t>(new uint64_t [dims.size()], [](uint64_t* p) { delete[] p; });
            coreml_layer_->upsample->scalingfactor = (uint64_t *) scalingfactor_.get();
            for(int i=0; i<dims.size(); i++){
                coreml_layer_->upsample->scalingfactor[i] = scales_[i];
            }
        }
    } else {
        //Note: CoreML infer output shape with formular below for intput shape, different from the one in upsample_layer.cc
        //CoreML: [C, H, W] -> [C, scalingFactor[0] * H, scalingFactor[1] * W]
        //TNN: [C, H, W] -> [C, round(scalingFactor[0] * H), round(scalingFactor[1] * W)]
        //Adjust scales
        for(int i=0; i<scales.size(); i++) {
            int input_dim = input_shape[input_shape_size - scales.size() + i];
            float output_dim = scales[scales.size() - i - 1] * input_dim;
            float output_dim_r = round(output_dim);
            if (output_dim_r - floor(output_dim) > 0) {
                scales[scales.size() - i - 1] = output_dim_r/input_dim;
            }
        }
        for(int i=0;i<scales.size();i++){
            if((scales[i]-((int)scales[i])) > 0.000001){
                isFractional = true;
            }
        }
        if(isFractional) {
            coreml_layer_->upsample->n_fractionalscalingfactor = scales.size();
            fractionalscalingfactor_ = std::shared_ptr<float>(new float [scales.size()], [](float* p) { delete[] p; });
            coreml_layer_->upsample->fractionalscalingfactor = (float *) fractionalscalingfactor_.get();
            for(int i=0; i<scales.size(); i++ ) {
                coreml_layer_->upsample->fractionalscalingfactor[i] = scales[scales.size() - i - 1];
            }
        } else {
            coreml_layer_->upsample->n_scalingfactor = scales.size();
            scalingfactor_ = std::shared_ptr<uint64_t>(new uint64_t [scales.size()], [](uint64_t* p) { delete[] p; });
            coreml_layer_->upsample->scalingfactor = (uint64_t *) scalingfactor_.get();
            for(int i=0; i<scales.size(); i++){
                coreml_layer_->upsample->scalingfactor[i] = scales[scales.size() - i - 1];
            }
        }
    }
    
    //CoreML only support integer scale for mode nn
    if (isFractional && mode == 1) {
        mode = 2;
    }
    
    if(mode == 1) {  // nearest
        coreml_layer_->upsample->mode = CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__INTERPOLATION_MODE__NN;
    } else if(mode == 2) {  // bilinear/linear
        coreml_layer_->upsample->mode = CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__INTERPOLATION_MODE__BILINEAR;
        // align corners option from pytorch
//        if(isFractional) { // Fractional upsample only compatible with align_corners=true or align_corners=false
            if(align_corners == 0) {// ALIGN_CORNERS_FALSE: spacing = Xin / Xout , grid_point[i] = min(Xin-1, max(0, i * spacing + 0.5 * spacing - 0.5)), for i = 0,1,2,….,Xout-1
                coreml_layer_->upsample->linearupsamplemode = CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__LINEAR_UPSAMPLE_MODE__ALIGN_CORNERS_FALSE;
            } else if(align_corners == 1) {// ALIGN_CORNERS_TRUE: spacing = (Xin-1) / (Xout-1) , grid_point[i] = min(Xin-1, max(0, i * spacing)), for i = 0,1,2,….,Xout-1
                coreml_layer_->upsample->linearupsamplemode=   CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__LINEAR_UPSAMPLE_MODE__ALIGN_CORNERS_TRUE;
            }
//        } else {
//            if(align_corners == 0) {// DEFAULT: spacing = (Xin-Xin/Xout) / (Xout-1) , grid_point[i] = min(Xin-1, max(0, i * spacing)), for i = 0,1,2,….,Xout-1
//                coreml_layer_->upsample->linearupsamplemode = CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__LINEAR_UPSAMPLE_MODE__DEFAULT;
//            } else if(align_corners == 1) {// ALIGN_CORNERS_TRUE: spacing = (Xin-1) / (Xout-1) , grid_point[i] = min(Xin-1, max(0, i * spacing)), for i = 0,1,2,….,Xout-1
//                coreml_layer_->upsample->linearupsamplemode=   CORE_ML__SPECIFICATION__UPSAMPLE_LAYER_PARAMS__LINEAR_UPSAMPLE_MODE__ALIGN_CORNERS_TRUE;
//            }
//        }
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
