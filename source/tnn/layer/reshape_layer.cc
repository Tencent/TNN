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
DECLARE_LAYER_WITH_FUNC(Reshape, LAYER_RESHAPE,
                        virtual Status FillLayerParamWithConstantResource(););

Status ReshapeLayer::InferOutputDataType() {
    BaseLayer::InferOutputDataType();
    
    auto layer_param = dynamic_cast<ReshapeLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (runtime_model_ == RUNTIME_MODE_CONST_FOLD) {
        for (auto& iter : output_blobs_) {
            int allocate_status = DATA_FLAG_ALLOCATE_IN_FORWARD;
            iter->SetFlag(iter->GetFlag() | allocate_status);
        }
    }
    
    return TNN_OK;
}

Status ReshapeLayer::InferOutputShape(bool ignore_error) {
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto layer_param = dynamic_cast<ReshapeLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];
    if (!layer_param->shape.empty()) {

        auto input_dims = input_blob->GetBlobDesc().dims;

        if (layer_param->num_axes == -1) {
            layer_param->num_axes = layer_param->shape.size();
        }

        Status status = TNN_OK;
        auto output_dims = DimsFunctionUtils::Reshape(input_dims, layer_param->shape, layer_param->axis, layer_param->num_axes, &status);
        RETURN_ON_NEQ(status, TNN_OK);
        
        output_blob->GetBlobDesc().dims = output_dims;
        return TNN_OK;
    } else {
        // shape is empty
        LOGE_IF(!ignore_error, "Reshape has no shape param. layer name: %s\n", layer_param->name.c_str());
        return Status(TNNERR_PARAM_ERR, "Reshape has no shape param");
    }
}

Status ReshapeLayer::FillLayerParamWithConstantResource() {
    Status status = TNN_OK;
    auto layer_param = dynamic_cast<ReshapeLayerParam *>(param_);
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
            layer_param->num_axes = dim_count;
        }
    }
    return status;
}

REGISTER_LAYER(Reshape, LAYER_RESHAPE);

}  // namespace TNN_NS
