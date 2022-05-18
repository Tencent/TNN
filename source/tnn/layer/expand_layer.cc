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

#include <cmath>

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_LAYER_WITH_FUNC(Expand, LAYER_EXPAND,
                        virtual Status FillLayerParamWithConstantResource(););

Status ExpandLayer::InferOutputDataType() {
    BaseLayer::InferOutputDataType();

    if (const_resource_) {
        const auto iter = const_resource_->find(input_blobs_[0]->GetBlobDesc().name);
        if (iter != const_resource_->end()) {
            output_blobs_[0]->GetBlobDesc().data_type = iter->second->GetDataType();
        }
    }

    return TNN_OK;
}

Status ExpandLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    auto layer_param = dynamic_cast<ExpandLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    Blob* input_blob = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];
    auto input_dims = input_blob->GetBlobDesc().dims;
    std::vector<int> shape_dims;
    shape_dims = layer_param->shape;
    
    auto output_dims = DimsFunctionUtils::Expand(input_dims, shape_dims, nullptr);
    output_blob->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

Status ExpandLayer::FillLayerParamWithConstantResource() {
    Status status = TNN_OK;
    auto layer_param = dynamic_cast<ExpandLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    //根据const resource更新维度信息
    if (input_blobs_.size() >= 2) {
        auto shape_blob_name = input_blobs_[1]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(shape_blob_name) != const_resource_->end()) {
            auto shape_buffer = (*const_resource_)[shape_blob_name];
            auto dim_count = shape_buffer->GetDataCount();
            auto dim_data = (int *)shape_buffer->force_to<int *>();
            DimsVector dims;
            for (int i=0; i<dim_count; i++) {
                dims.push_back(dim_data[i]);
            }
            layer_param->shape = dims;
        }
    }
    return status;
}

REGISTER_LAYER(Expand, LAYER_EXPAND);

}  // namespace TNN_NS
