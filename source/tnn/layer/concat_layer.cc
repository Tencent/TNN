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
#include "tnn/core/common.h"
#include "tnn/layer/base_layer.h"

namespace TNN_NS {

DECLARE_LAYER(Concat, LAYER_CONCAT);

inline bool ConcatLayerCheckShape(DimsVector shape1, DimsVector shape2, int exclude_axis) {
    if (shape1.size() != shape2.size()) {
        LOGE("shape1 dim size %d  shape2 dim size %d\n", (int)shape1.size(), (int)shape2.size());
        return false;
    }

    int i = 0;
    for (; i < shape1.size(); i++) {
        if (i != exclude_axis && shape1[i] != shape2[i]) {
            LOGE("dim[%d] not match (shape1:%d, shape2:%d)\n", i, shape1[i], shape2[i]);
            return false;
        }
    }
    return true;
}

Status ConcatLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status ConcatLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    ConcatLayerParam* concat_param = dynamic_cast<ConcatLayerParam*>(param_);
    CHECK_PARAM_NULL(concat_param);

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    const int axis = concat_param->axis;
    if (axis < 0 || axis > input_blob->GetBlobDesc().dims.size()) {
        LOGE_IF(!ignore_error, "Error: ConcatLayer axis(%d) is invalid\n", axis);
        return Status(TNNERR_PARAM_ERR, "ConcatLayer axis is invalid");
    }

    size_t i                = 0;
    auto last_shape         = input_blobs_[i]->GetBlobDesc().dims;
    int out_concat_dim_size = 0;
    for (; i < input_blobs_.size(); i++) {
        auto input_blob = input_blobs_[i];
        auto cur_shape  = input_blob->GetBlobDesc().dims;
        if (!ConcatLayerCheckShape(last_shape, cur_shape, axis)) {
            LOGE_IF(!ignore_error,
                "Error: ConcatLayer's (layer name: %s) inputs can not be "
                "concatenated with "
                "axis=%d\n",
                GetLayerName().c_str(), axis);
            return Status(TNNERR_PARAM_ERR, "ConcatLayer's inputs can not be concatenated");
        }
        out_concat_dim_size += cur_shape[axis];
    }

    last_shape[axis] = out_concat_dim_size;

    output_blob->GetBlobDesc().dims = last_shape;

    return TNN_OK;
}

REGISTER_LAYER(Concat, LAYER_CONCAT);

}  // namespace TNN_NS
