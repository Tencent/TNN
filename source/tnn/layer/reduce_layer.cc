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

#include "reduce_layer.h"

#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

Status ReduceLayer::InferOutputShape() {
    auto layer_param = dynamic_cast<ReduceLayerParam*>(param_);
    if (!layer_param) {
        LOGE("Error: Reduce may not support axes != 1, depend on device\n");
        return Status(TNNERR_MODEL_ERR, "Error: Reduce may not support axes != 1, depend on device");
    }

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];
    auto dims         = input_blob->GetBlobDesc().dims;

    for (auto& axis : layer_param->axis) {
        axis = axis >= 0 ? axis : axis + (int)dims.size();
        if (axis < 0 || axis >= dims.size()) {
            LOGE("Error: layer param axis is invalid\n");
            return Status(TNNERR_MODEL_ERR, "Error: layer param axis is invalid");
        }
        dims[axis] = 1;
    }
    output_blob->GetBlobDesc().dims = dims;

    return TNN_OK;
}
}  // namespace TNN_NS