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
DECLARE_LAYER_WITH_FUNC(Pow, LAYER_POWER,
                        virtual Status FillLayerParamWithConstantResource(););

Status PowLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status PowLayer::InferOutputShape(bool ignore_error) {
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto layer_param = dynamic_cast<PowLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    output_blob->GetBlobDesc().dims = input_blob->GetBlobDesc().dims;
    return TNN_OK;
}

Status PowLayer::FillLayerParamWithConstantResource() {
    Status status = TNN_OK;
    auto *layer_param = dynamic_cast<PowLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (input_blobs_.size() >= 2) {
        auto min_blob_name = input_blobs_[1]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(min_blob_name) != const_resource_->end()) {
            auto min_buffer =  (*const_resource_)[min_blob_name];
            auto dim_count = min_buffer->GetDataCount();
            if (min_buffer->GetDataType() == DATA_TYPE_FLOAT) {
                auto dim_data = (float *)min_buffer->force_to<float *>();
                layer_param->exponent = *dim_data;
            } else {
                return Status(TNNERR_PARAM_ERR, "ClipLayer has invalid data type for min value");
            }
            
            if (dim_count > 1) {
                return Status(TNNERR_PARAM_ERR, "PowLayer only dont support broad cast right now");
            }
        }
    }
    return status;
}
REGISTER_ELEMENTWISE_LAYER(Pow, LAYER_POWER);

}  // namespace TNN_NS
