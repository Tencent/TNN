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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
DECLARE_LAYER(Where, LAYER_WHERE);

Status WhereLayer::InferOutputDataType() {
    auto status = BaseLayer::InferOutputDataType();
    RETURN_ON_NEQ(status, TNN_OK);

    auto layer_resource = dynamic_cast<WhereLayerResource*>(resource_);
    if (layer_resource) {
        if (layer_resource->x.GetBytesSize()>0 && layer_resource->y.GetBytesSize()>0) {
            if (layer_resource->x.GetDataType()!=layer_resource->y.GetDataType()) {
                return Status(TNNERR_PARAM_ERR, "DataType WhereTorchLayer x(Constant) and y(Constant) should be the same: " + layer_name_);
            } else {
                output_blobs_[0]->GetBlobDesc().data_type = layer_resource->x.GetDataType();
            }
        }
        // If at least one of x and y is not stored in LayerResouce.
        // The first input of Where Layer should be the remaining x or y. Out DataType==in0.data_type.
    }
 
    output_blobs_[0]->GetBlobDesc().data_type = input_blobs_[0]->GetBlobDesc().data_type;
    return TNN_OK;
}

Status WhereLayer::InferOutputShape(bool ignore_error) {
    //X, Y, condition order for input
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);
    
    Blob* input_blob = input_blobs_[0];
    auto dims = input_blob->GetBlobDesc().dims;
    auto dims_output = dims;
    for (auto iter : input_blobs_) {
        dims       = iter->GetBlobDesc().dims;
        dims_output = DimsVectorUtils::Max(dims, dims_output);
    }

    // For Where Torch, x or y may be in resource instead of in inputs.
    auto layer_resource = dynamic_cast<WhereLayerResource*>(resource_);
    if (layer_resource) {
        if (layer_resource->x.GetBytesSize()>0) {
            dims = layer_resource->x.GetBufferDims();
            dims_output = DimsVectorUtils::Max(dims, dims_output);
        }
        if (layer_resource->y.GetBytesSize()>0) {
            dims = layer_resource->y.GetBufferDims();
            dims_output = DimsVectorUtils::Max(dims, dims_output);
        }
    }

    output_blobs_[0]->GetBlobDesc().dims = dims_output;
    return TNN_OK;
}
REGISTER_LAYER(Where, LAYER_WHERE);

}  // namespace TNN_NS
