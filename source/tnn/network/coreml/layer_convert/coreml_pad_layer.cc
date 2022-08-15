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

DECLARE_COREML_LAYER_WITH_DATA(Pad, LAYER_PAD,
                                std::shared_ptr<void> pad_type_;
                                std::shared_ptr<void> paddingamounts_;
                                std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes*> borderamounts_;
                                std::vector<std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes> > borderamounts_arr_;);

Status CoreMLPadLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_PADDING;
    return TNN_OK;
}

Status CoreMLPadLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<PadLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    // for old Pad the order is  [w_begin, w_end, h_begin, h_end, c_begin, c_end]
    auto pads = layer_param->pads;
    auto w_begin = pads[0];
    auto w_end = pads[1];
    auto h_begin = pads[2];
    auto h_end = pads[3];
    auto c_begin = pads[4];
    auto c_end = pads[5];
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
    
    if ((c_begin || c_end) && (type == 0)){ // constant padding, allowed for C , H and W dimensions
        coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_CONSTANT_PAD;
        coreml_layer_param_ = std::shared_ptr<CoreML__Specification__ConstantPaddingLayerParams>(new CoreML__Specification__ConstantPaddingLayerParams);
        coreml_layer_->constantpad = (CoreML__Specification__ConstantPaddingLayerParams *)coreml_layer_param_.get();
        core_ml__specification__constant_padding_layer_params__init(coreml_layer_->constantpad);
        coreml_layer_->constantpad->value = value;
        coreml_layer_->constantpad->n_padamounts = input_shape_size*2;
        paddingamounts_ = std::shared_ptr<uint64_t*>(new uint64_t* [input_shape_size*2], [](uint64_t** p) { delete[] p; });
        coreml_layer_->constantpad->padamounts = (uint64_t *) paddingamounts_.get();
        
        for(int i=0;i<2;i++){ // add N dim pad n1 n2 (=0)
            pads.push_back(0);
        }
        for(int i=0; i<input_shape_size*2; i+=2){
            coreml_layer_->constantpad->padamounts[i] = pads[input_shape_size*2-i-2];
            coreml_layer_->constantpad->padamounts[i+1] = pads[input_shape_size*2-i-1];
        }
    } else if ((c_begin == 0) && (c_end == 0)) { // three types of padding, only allowed for H and W dimensions
        coreml_layer_param_ = std::shared_ptr<CoreML__Specification__PaddingLayerParams>(new CoreML__Specification__PaddingLayerParams);
        coreml_layer_->padding = (CoreML__Specification__PaddingLayerParams *)coreml_layer_param_.get();
        core_ml__specification__padding_layer_params__init(coreml_layer_->padding);
        
        paddingamounts_ = std::shared_ptr<CoreML__Specification__BorderAmounts>(new CoreML__Specification__BorderAmounts);
        coreml_layer_->padding->paddingamounts = (CoreML__Specification__BorderAmounts *)paddingamounts_.get();
        core_ml__specification__border_amounts__init(coreml_layer_->padding->paddingamounts);
        coreml_layer_->padding->paddingamounts->n_borderamounts = 2;
        borderamounts_ = std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes*>(new CoreML__Specification__BorderAmounts__EdgeSizes* [2], [](CoreML__Specification__BorderAmounts__EdgeSizes** p) { delete[] p; });
        coreml_layer_->padding->paddingamounts->borderamounts = borderamounts_.get();
        borderamounts_arr_.push_back(std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes>(new CoreML__Specification__BorderAmounts__EdgeSizes));
        coreml_layer_->padding->paddingamounts->borderamounts[0] = borderamounts_arr_[0].get();
        core_ml__specification__border_amounts__edge_sizes__init(coreml_layer_->padding->paddingamounts->borderamounts[0]);
        // the order ``[H, W]``.
        coreml_layer_->padding->paddingamounts->borderamounts[0]->startedgesize = h_begin;
        coreml_layer_->padding->paddingamounts->borderamounts[0]->endedgesize = h_end;
        borderamounts_arr_.push_back(std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes>(new CoreML__Specification__BorderAmounts__EdgeSizes));
        coreml_layer_->padding->paddingamounts->borderamounts[1] = borderamounts_arr_[1].get();
        core_ml__specification__border_amounts__edge_sizes__init(coreml_layer_->padding->paddingamounts->borderamounts[1]);
        coreml_layer_->padding->paddingamounts->borderamounts[1]->startedgesize = w_begin;
        coreml_layer_->padding->paddingamounts->borderamounts[1]->endedgesize = w_end;
        
        /* There are three types of padding:
        * - ``PaddingConstant``, which fills a constant value at the border.
        * - ``PaddingReflection``, which reflects the values at the border.
        * - ``PaddingReplication``, which replicates the values at the border. */
        if (type == 0) {
            coreml_layer_->padding->padding_type_case = CORE_ML__SPECIFICATION__PADDING_LAYER_PARAMS__PADDING_TYPE_CONSTANT;
            pad_type_ = std::shared_ptr<CoreML__Specification__PaddingLayerParams__PaddingConstant>(new CoreML__Specification__PaddingLayerParams__PaddingConstant);
            coreml_layer_->padding->constant = (CoreML__Specification__PaddingLayerParams__PaddingConstant *) pad_type_.get();
            core_ml__specification__padding_layer_params__padding_constant__init(coreml_layer_->padding->constant);
            coreml_layer_->padding->constant->value = value;
        } else if (type == 1) {
            coreml_layer_->padding->padding_type_case = CORE_ML__SPECIFICATION__PADDING_LAYER_PARAMS__PADDING_TYPE_REFLECTION;
            pad_type_ = std::shared_ptr<CoreML__Specification__PaddingLayerParams__PaddingReflection>(new CoreML__Specification__PaddingLayerParams__PaddingReflection);
            coreml_layer_->padding->reflection = (CoreML__Specification__PaddingLayerParams__PaddingReflection *) pad_type_.get();
            core_ml__specification__padding_layer_params__padding_reflection__init(coreml_layer_->padding->reflection);
        } else if (type == 2) {
            coreml_layer_->padding->padding_type_case = CORE_ML__SPECIFICATION__PADDING_LAYER_PARAMS__PADDING_TYPE_REPLICATION;
            pad_type_ = std::shared_ptr<CoreML__Specification__PaddingLayerParams__PaddingReplication>(new CoreML__Specification__PaddingLayerParams__PaddingReplication);
            coreml_layer_->padding->replication = (CoreML__Specification__PaddingLayerParams__PaddingReplication *) pad_type_.get();
            core_ml__specification__padding_layer_params__padding_replication__init(coreml_layer_->padding->replication);
        }
    } else {
        return Status(TNNERR_MODEL_ERR, "CoreMLPadLayer only allowed constant padding for C, H and W dimensions & three types of padding for H and W dimensions");
    }
    
    return TNN_OK;
}

Status CoreMLPadLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLPadLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLPadLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Pad, LAYER_PAD);

}  // namespace TNN_NS
