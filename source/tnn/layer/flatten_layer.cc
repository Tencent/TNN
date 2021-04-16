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

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_LAYER(Flatten, LAYER_FLATTEN);

Status FlattenLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status FlattenLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    FlattenLayerParam* flatten_param = dynamic_cast<FlattenLayerParam*>(param_);
    CHECK_PARAM_NULL(flatten_param);

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    const int axis            = flatten_param->axis;
    const auto input_dims     = input_blob->GetBlobDesc().dims;
    const int input_dims_size = input_dims.size();
    if (axis < 0 || axis > input_dims_size) {
        LOGE_IF(!ignore_error, "flatten param size error\n");
        return Status(TNNERR_PARAM_ERR, "flatten param size error");
    }

    int dim0 = 1;
    int dim1 = 1;
    for (int i = 0; i < axis; i++) {
        dim0 *= input_dims[i];
    }
    for (int i = axis; i < input_dims_size; i++) {
        dim1 *= input_dims[i];
    }

    output_blob->GetBlobDesc().dims = {dim0, dim1};

    return TNN_OK;
}

REGISTER_LAYER(Flatten, LAYER_FLATTEN);

}  // namespace TNN_NS
