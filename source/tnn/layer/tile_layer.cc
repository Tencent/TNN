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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_LAYER_WITH_FUNC(Tile, LAYER_REPEAT,
                        virtual Status FillLayerParamWithConstantResource(););

Status TileLayer::InferOutputDataType() {
    auto status = BaseLayer::InferOutputDataType();
    RETURN_ON_NEQ(status, TNN_OK);

    output_blobs_[0]->GetBlobDesc().data_type = input_blobs_[0]->GetBlobDesc().data_type;

    return TNN_OK;
}

Status TileLayer::InferOutputShape(bool ignore_error) {
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto *layer_param = dynamic_cast<TileLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];
    
    auto input_dims = input_blob->GetBlobDesc().dims;
    auto reps = layer_param->reps;
    auto output_dims = DimsFunctionUtils::Tile(input_dims, reps);

    output_blob->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

Status TileLayer::FillLayerParamWithConstantResource() {
    Status status = TNN_OK;
    auto *layer_param = dynamic_cast<TileLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (input_blobs_.size() < 2) {
        return TNN_OK;
    }
    
    //reps
    {
        const auto name = input_blobs_[1]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(name) != const_resource_->end()) {
            auto buffer = (*const_resource_)[name];
            if (buffer->GetDataType() != DATA_TYPE_INT32) {
                return Status(TNNERR_PARAM_ERR, "TileLayer has invalid reps data type");
            }
            auto dim_count = buffer->GetDataCount();
            auto dim_data = (int *)buffer->force_to<int *>();
            DimsVector dims;
            for (int i=0; i<dim_count; i++) {
                dims.push_back(dim_data[i]);
            }
            layer_param->reps = dims;
        }
    }
    
    return status;
}

REGISTER_LAYER(Tile, LAYER_REPEAT);

}  // namespace TNN_NS
