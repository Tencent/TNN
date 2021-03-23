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

#include "tnn/layer/elementwise_layer.h"

namespace TNN_NS {
DECLARE_LAYER_WITH_FUNC(Clip, LAYER_CLIP,
                        virtual Status FillLayerParamWithConstantResource(););

Status ClipLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status ClipLayer::InferOutputShape(bool ignore_error) {
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto layer_param = dynamic_cast<ClipLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    output_blob->GetBlobDesc().dims = input_blob->GetBlobDesc().dims;
    return TNN_OK;
}

Status ClipLayer::FillLayerParamWithConstantResource() {
    Status status = TNN_OK;
    auto *layer_param = dynamic_cast<ClipLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (input_blobs_.size() >= 2) {
        auto min_blob_name = input_blobs_[1]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(min_blob_name) != const_resource_->end()) {
            auto min_buffer =  (*const_resource_)[min_blob_name];
            auto dim_count = min_buffer->GetDataCount();
            if (min_buffer->GetDataType() == DATA_TYPE_FLOAT) {
                auto dim_data = (float *)min_buffer->force_to<float *>();
                layer_param->min = *dim_data;
            } else {
                return Status(TNNERR_PARAM_ERR, "ClipLayer has invalid data type for min value");
            }
        }
    }
    
    if (input_blobs_.size() >= 3) {
        auto max_blob_name = input_blobs_[2]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(max_blob_name) != const_resource_->end()) {
            auto max_buffer =  (*const_resource_)[max_blob_name];
            auto dim_count = max_buffer->GetDataCount();
            if (max_buffer->GetDataType() == DATA_TYPE_FLOAT) {
                auto dim_data = (float *)max_buffer->force_to<float *>();
                layer_param->max = *dim_data;
            } else {
                return Status(TNNERR_PARAM_ERR, "ClipLayer has invalid data type for min value");
            }
        }
    }
    
    return status;
}

REGISTER_ELEMENTWISE_LAYER(Clip, LAYER_CLIP);

}  // namespace TNN_NS
