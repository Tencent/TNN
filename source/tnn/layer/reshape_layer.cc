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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
DECLARE_LAYER(Reshape, LAYER_RESHAPE);

Status ReshapeLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status ReshapeLayer::InferOutputShape() {
    ReshapeLayerParam* reshape_param = dynamic_cast<ReshapeLayerParam*>(param_);
    CHECK_PARAM_NULL(reshape_param);

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    if ((reshape_param->shape.size() + reshape_param->axis) != input_blob->GetBlobDesc().dims.size()) {
        LOGE("reshape param size error\n");
        return Status(TNNERR_PARAM_ERR, "reshape param size error");
    }

    auto output_dims = input_blob->GetBlobDesc().dims;

    int infer_dim_count = 0;
    int infer_dim_pos   = -1;
    for (int i = reshape_param->axis, j = 0; i < reshape_param->num_axes; i++, j++) {
        if (reshape_param->shape[j] == -1) {
            infer_dim_count += 1;
            infer_dim_pos  = i;
            output_dims[i] = 1;
        } else if (reshape_param->shape[j] == 0) {
            output_dims[i] = input_blob->GetBlobDesc().dims[i];
        } else {
            output_dims[i] = reshape_param->shape[j];
        }
    }

    output_blob->GetBlobDesc().dims = output_dims;
    // temporary fix reshpae init error
    if (infer_dim_count == 0 && infer_dim_pos == -1) {
        return TNN_OK;
    }

    if (infer_dim_count != 1 || infer_dim_pos == -1) {
        LOGE("reshape param size error\n");
        return Status(TNNERR_PARAM_ERR, "reshape param size error");
    }

    int in_cnt  = DimsVectorUtils::Count(input_blob->GetBlobDesc().dims);
    int out_cnt = DimsVectorUtils::Count(output_dims);
    if (0 == out_cnt) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    output_dims[infer_dim_pos]      = in_cnt / out_cnt;
    output_blob->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(Reshape, LAYER_RESHAPE);

}  // namespace TNN_NS
