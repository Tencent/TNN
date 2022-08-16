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

DECLARE_COREML_LAYER_WITH_DATA(PadV2, LAYER_PADV2,
                                std::shared_ptr<void> pad_type_;
                                std::shared_ptr<void> paddingamounts_;
                                std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes*> borderamounts_;
                                std::vector<std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes> > borderamounts_arr_;);

Status CoreMLPadV2Layer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_PADDING;
    return TNN_OK;
}

Status CoreMLPadV2Layer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<PadLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    
    auto pads = layer_param->pads;
    
    // 0:const 1:reflect 2:edge
    auto type = layer_param->type;
    auto value = layer_param->value;
    
    int input_shape_size = 0;
    if (net_resource_ && layer_info_->inputs.size()>0 && layer_info_->outputs.size()>0) {
        if (net_resource_->blob_shapes_map.find(layer_info_->inputs[0]) != net_resource_->blob_shapes_map.end()) {
            auto input_shape = net_resource_->blob_shapes_map[layer_info_->inputs[0]];
            input_shape_size = (int)input_shape.size();
        }
    }
    if (input_shape_size == 0) {
        input_shape_size = pads.size()/2;
    }
    
    if (input_shape_size*2 != pads.size()) {
        return Status(TNNERR_MODEL_ERR, "CoreMLPadV2Layer has invalid input dim size");
    }
    
    if (type == 0){ // constant padding, allowed for C , H and W dimensions
        coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_CONSTANT_PAD;
        coreml_layer_param_ = std::shared_ptr<CoreML__Specification__ConstantPaddingLayerParams>(new CoreML__Specification__ConstantPaddingLayerParams);
        coreml_layer_->constantpad = (CoreML__Specification__ConstantPaddingLayerParams *)coreml_layer_param_.get();
        core_ml__specification__constant_padding_layer_params__init(coreml_layer_->constantpad);
        coreml_layer_->constantpad->value = value;
        coreml_layer_->constantpad->n_padamounts = input_shape_size*2;
        paddingamounts_ = std::shared_ptr<uint64_t*>(new uint64_t* [input_shape_size*2], [](uint64_t** p) { delete[] p; });
        coreml_layer_->constantpad->padamounts = (uint64_t *) paddingamounts_.get();
     
        for(int i=0; i<input_shape_size; i++){
            coreml_layer_->constantpad->padamounts[2*i] = pads[i];
            coreml_layer_->constantpad->padamounts[2*i+1] = pads[i + input_shape_size];
        }
    } else {
        return Status(TNNERR_MODEL_ERR, "CoreMLPadV2Layer only allowed constant padding");
    }
    
    return TNN_OK;
}

Status CoreMLPadV2Layer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLPadV2Layer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLPadV2Layer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(PadV2, LAYER_PADV2);

}  // namespace TNN_NS
