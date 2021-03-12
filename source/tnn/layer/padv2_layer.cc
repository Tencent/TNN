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
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {
DECLARE_LAYER_WITH_FUNC(PadV2, LAYER_PADV2,
                        virtual Status FillLayerParamWithConstantResource(););
Status PadV2Layer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status PadV2Layer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    auto layer_param = dynamic_cast<PadLayerParam*>(param_);
    if (!layer_param) {
        LOGE_IF(!ignore_error, "Error: layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: layer param is nil");
    }

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];
    auto output_dims = input_blob->GetBlobDesc().dims;
    auto dim_size = layer_param->pads.size()/2;
    dim_size = dim_size <= output_dims.size() ? dim_size : output_dims.size();
    for (int i = 0; i<dim_size; i++) {
        output_dims[i] += layer_param->pads[i] + layer_param->pads[i+dim_size];
    }
    
    output_blob->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

Status PadV2Layer::FillLayerParamWithConstantResource() {
    Status status = TNN_OK;
    auto *layer_param = dynamic_cast<PadLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (input_blobs_.size() >= 2) {
        auto pads_blob_name = input_blobs_[1]->GetBlobDesc().name;
        if (const_resource_ != nullptr && const_resource_->find(pads_blob_name) != const_resource_->end()) {
            auto begins_buffer =  (*const_resource_)[pads_blob_name];
            auto dim_count = begins_buffer->GetDataCount();
            if (begins_buffer->GetDataType() == DATA_TYPE_INT32) {
                auto dim_data = (int *)begins_buffer->force_to<int *>();
                DimsVector dims;
                for (int i=0; i<dim_count; i++) {
                    dims.push_back(dim_data[i]);
                }
                layer_param->pads = dims;
            } else if(begins_buffer->GetDataType() == DATA_TYPE_INT64){
                auto dim_data = (long long int *)begins_buffer->force_to<long long int *>();
                DimsVector dims;
                for (int i=0; i<dim_count; i++) {
                    dims.push_back(DataTypeUtils::SaturateCast(dim_data[i]));
                }
                layer_param->pads = dims;
            }
        }
    }
    
    return status;
}

REGISTER_LAYER(PadV2, LAYER_PADV2);

}  // namespace TNN_NS
