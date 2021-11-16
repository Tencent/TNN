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

#include "base_layer.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
DECLARE_LAYER(Greater, LAYER_GREATER);

Status GreaterLayer::InferOutputDataType() {
    BaseLayer::InferOutputDataType();
    for (auto output_blob : output_blobs_) {
        output_blob->GetBlobDesc().data_type = DATA_TYPE_INT8;
    }
    return TNN_OK;
}

Status GreaterLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    Blob* input_blob = input_blobs_[0];

    auto dims0       = input_blob->GetBlobDesc().dims;
    auto dims_output = dims0;
    for (auto iter : input_blobs_) {
        dims0       = iter->GetBlobDesc().dims;
        dims_output = DimsVectorUtils::Max(dims0, dims_output);
    }

    output_blobs_[0]->GetBlobDesc().dims = dims_output;
    return TNN_OK;
}

REGISTER_LAYER(Greater, LAYER_GREATER);

}  // namespace TNN_NS
