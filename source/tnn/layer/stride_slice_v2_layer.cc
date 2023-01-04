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

#include <algorithm>

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_LAYER_WITH_FUNC(StrideSliceV2, LAYER_STRIDED_SLICE_V2,
                        virtual Status FillLayerParamWithConstantResource(););

Status StrideSliceV2Layer::InferOutputDataType() {
    auto status = BaseLayer::InferOutputDataType();
    RETURN_ON_NEQ(status, TNN_OK);

    output_blobs_[0]->GetBlobDesc().data_type = input_blobs_[0]->GetBlobDesc().data_type;

    return TNN_OK;
}

Status StrideSliceV2Layer::InferOutputShape(bool ignore_error) {
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto layer_param = dynamic_cast<StrideSliceV2LayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    auto input_dims = input_blob->GetBlobDesc().dims;

    auto& begins = layer_param->begins;
    auto& ends = layer_param->ends;
    auto& axes = layer_param->axes;
    auto& strides = layer_param->strides;

    // prepare process begin and ends
    auto output_dims = DimsFunctionUtils::StrideSlice(input_dims, begins, ends, strides, axes, &status);
    //support empty blob for yolov5 Slice_507, only in device cpu
    if (status != TNN_OK && !(output_dims.size() == input_dims.size() &&  runtime_model_ == RUNTIME_MODE_CONST_FOLD)) {
        return status;
    }
  
    //dont rectify begins and ends here, input shape may change, do it in runtime forward see cpu_stride_slice_v2_layer_acc.cc Forward
//    layer_param->begins = begins;
//    layer_param->ends = ends;
    output_blob->GetBlobDesc().dims = output_dims;

    return TNN_OK;
}

Status StrideSliceV2Layer::FillLayerParamWithConstantResource() {
    Status status = TNN_OK;
    auto *layer_param = dynamic_cast<StrideSliceV2LayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (input_blobs_.size() >= 2) {
        auto begins_blob_name = input_blobs_[1]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(begins_blob_name) != const_resource_->end()) {
            auto begins_buffer =  (*const_resource_)[begins_blob_name];
            auto dim_count = begins_buffer->GetDataCount();
            auto dim_data = (int *)begins_buffer->force_to<int *>();
            DimsVector dims;
            for (int i=0; i<dim_count; i++) {
                dims.push_back(dim_data[i]);
            }
            layer_param->begins = dims;
        }
    }
    
    if (input_blobs_.size() >= 3) {
        auto ends_blob_name = input_blobs_[2]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(ends_blob_name) != const_resource_->end()) {
            auto ends_buffer =  (*const_resource_)[ends_blob_name];
            auto dim_count = ends_buffer->GetDataCount();
            auto dim_data = (int *)ends_buffer->force_to<int *>();
            DimsVector dims;
            for (int i=0; i<dim_count; i++) {
                dims.push_back(dim_data[i]);
            }
            layer_param->ends = dims;
        }
    }
    
    return status;
}

REGISTER_LAYER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

}  // namespace TNN_NS
