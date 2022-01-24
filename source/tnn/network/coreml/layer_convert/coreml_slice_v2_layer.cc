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
DECLARE_COREML_LAYER_WITH_FUNC_DATA(SliceV2, LAYER_STRIDED_SLICE_V2,
                                    bool IsDynamic();,
                                    std::shared_ptr<void> coreml_layer_type_;
                                    std::shared_ptr<int64_t> coreml_layer_begins_;
                                    std::shared_ptr<int> coreml_layer_begin_masks_;
                                    std::shared_ptr<int64_t> coreml_layer_ends_;
                                    std::shared_ptr<int> coreml_layer_end_masks_;
                                    std::shared_ptr<int64_t> coreml_layer_strides_;
                                    std::shared_ptr<int> coreml_layer_suqeeze_masks_;);

bool CoreMLSliceV2Layer::IsDynamic() {
    if (layer_info_ && net_resource_ && layer_info_->inputs.size() == 1) {
        if (net_resource_->constant_map.find(layer_info_->inputs[0]) == net_resource_->constant_map.end()) {
            return false;
        }
    }
    return true;
}

Status CoreMLSliceV2Layer::BuildLayerType() {
    //layer type
    if (IsDynamic()) {
        coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SLICE_DYNAMIC;
    } else {
        //use slicestatic not slice, slice dont work for case with axis = 0 and input shape [2, 2, 4]
        //coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SCALE;
        coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_SLICE_STATIC;
    }
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
    const int input_shape_size = (int)input_shape.size();
    if (input_shape_size == 0) {
        return Status(TNNERR_COMMON_ERROR, "CoreMLSliceV2Layer has invalid input shape size");
    }
    
    if (IsDynamic()) {
        coreml_layer_param_ = std::shared_ptr<CoreML__Specification__SliceDynamicLayerParams>(new CoreML__Specification__SliceDynamicLayerParams);
        coreml_layer_->slicedynamic = (CoreML__Specification__SliceDynamicLayerParams *)coreml_layer_param_.get();
        core_ml__specification__slice_dynamic_layer_params__init(coreml_layer_->slicedynamic);
        
    } else {
        coreml_layer_param_ = std::shared_ptr<CoreML__Specification__SliceStaticLayerParams>(new CoreML__Specification__SliceStaticLayerParams);
        coreml_layer_->slicestatic = (CoreML__Specification__SliceStaticLayerParams *)coreml_layer_param_.get();
        core_ml__specification__slice_static_layer_params__init(coreml_layer_->slicestatic);
        
        coreml_layer_->slicestatic->n_beginids = input_shape_size;
        coreml_layer_->slicestatic->n_endids = input_shape_size;
        coreml_layer_->slicestatic->n_strides = input_shape_size;
//        coreml_layer_->slicestatic->n_beginmasks = input_shape_size;
//        coreml_layer_->slicestatic->n_endmasks = input_shape_size;
//        coreml_layer_->slicestatic->n_squeezemasks = input_shape_size;
        
        coreml_layer_begins_ = std::shared_ptr<int64_t>(new int64_t [input_shape_size], [](int64_t* p) { delete[] p; });
        coreml_layer_ends_ = std::shared_ptr<int64_t>(new int64_t [input_shape_size], [](int64_t* p) { delete[] p; });
        coreml_layer_strides_ = std::shared_ptr<int64_t>(new int64_t [input_shape_size], [](int64_t* p) { delete[] p; });
//        coreml_layer_begin_masks_ = std::shared_ptr<int>(new int [input_shape_size], [](int* p) { delete[] p; });
//        coreml_layer_end_masks_ = std::shared_ptr<int>(new int [input_shape_size], [](int* p) { delete[] p; });
//        coreml_layer_suqeeze_masks_ = std::shared_ptr<int>(new int [input_shape_size], [](int* p) { delete[] p; });
        
        auto coreml_layer_begins_ptr = coreml_layer_begins_.get();
        auto coreml_layer_ends_ptr = coreml_layer_ends_.get();
        auto coreml_layer_strides_ptr = coreml_layer_strides_.get();
//        auto coreml_layer_begin_masks_ptr = coreml_layer_begin_masks_.get();
//        auto coreml_layer_end_masks_ptr = coreml_layer_end_masks_.get();
//        auto coreml_layer_suqeeze_masks_ptr = coreml_layer_suqeeze_masks_.get();
        
        //set default value
        for (int index = 0; index < input_shape_size; index++) {
            coreml_layer_begins_ptr[index] = 0;
            coreml_layer_ends_ptr[index] = -1;
            coreml_layer_strides_ptr[index] = 1;
//            coreml_layer_begin_masks_ptr[index] = 0;
//            coreml_layer_end_masks_ptr[index] = 0;
//            coreml_layer_suqeeze_masks_ptr[index] = 0;
        }
        
        for (int index = 0; index < axes.size(); index++) {
            auto axis = axes[index];
            coreml_layer_begins_ptr[axis] = begins[index];
            coreml_layer_ends_ptr[axis] = ends[index];
            coreml_layer_strides_ptr[axis] = strides[index];
        }
        
        coreml_layer_->slicestatic->beginids = coreml_layer_begins_ptr;
        coreml_layer_->slicestatic->endids = coreml_layer_ends_ptr;
        coreml_layer_->slicestatic->strides = coreml_layer_strides_ptr;
//        coreml_layer_->slicestatic->beginmasks = coreml_layer_begin_masks_ptr;
//        coreml_layer_->slicestatic->endmasks = coreml_layer_end_masks_ptr;
//        coreml_layer_->slicestatic->squeezemasks = coreml_layer_suqeeze_masks_ptr;
    }
    
    return TNN_OK;
}

Status CoreMLSliceV2Layer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLSliceV2Layer::BuildLayerInputs() {
    if (IsDynamic()) {
        return layer_info_->inputs;
    } else {
        return {layer_info_->inputs[0]};
    }
}

std::vector<std::string> CoreMLSliceV2Layer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(SliceV2, LAYER_STRIDED_SLICE_V2);

}  // namespace TNN_NS
