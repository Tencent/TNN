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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_LAYER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

Status StrideSliceV2Layer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status StrideSliceV2Layer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    StrideSliceV2LayerParam* layer_param = dynamic_cast<StrideSliceV2LayerParam*>(param_);
    if (!layer_param) {
        LOGE_IF(!ignore_error, "StrideSliceV2Layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "StrideSliceV2Layer param is nil");
    }
    
    //根据const resource更新维度信息
    if (runtime_model_ == RUNTIME_MODE_NORMAL) {
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
    }

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    auto input_dims = input_blob->GetBlobDesc().dims;

    auto begins = layer_param->begins;
    auto ends = layer_param->ends;
    auto axes = layer_param->axes;
    auto strides = layer_param->strides;

    //前闭后开区间
    Status status = TNN_OK;
    auto output_dims = DimsVectorUtils::StrideSlice(input_dims, begins, ends, strides, axes, &status);
    RETURN_ON_NEQ(status, TNN_OK);
  
    //dont rectify begins and ends here, input shape may change, do it in runtime forword see cpu_stride_slice_v2_layer_acc.cc Forword
//    layer_param->begins = begins;
//    layer_param->ends = ends;
    output_blob->GetBlobDesc().dims = output_dims;

    return TNN_OK;
}

REGISTER_LAYER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

}  // namespace TNN_NS
