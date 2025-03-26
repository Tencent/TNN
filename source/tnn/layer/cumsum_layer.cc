// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

namespace TNN_NS {

DECLARE_LAYER(Cumsum, LAYER_CUMSUM);

// ONNX Cumsum compatible with torch cumsum
// Plus an extra mode: Exclusive Extend
// Example:
// input_x = [1, 2, 3]
// axis=0
//
// Default:
// output  = [1, 3, 6]
//
// exclusive = 1
// output  = [0, 1, 3]
//
// exclusive_extend = 1
// output  = [0, 1, 3, 6]
//
// reverse = 1
// output  = [6, 5, 3]
//
// reverse = 1, exclusive = 1
// output  = [5, 3, 0]
//
// reverse = 1, exclusive_extend = 1
// output  = [5, 5, 3, 0]


Status CumsumLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status CumsumLayer::InferOutputShape(bool ignore_error) {
    auto status = BaseLayer::InferOutputShape(ignore_error);
    RETURN_ON_NEQ(status, TNN_OK);

    auto layer_param = dynamic_cast<CumsumLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);

    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    output_blob->GetBlobDesc().dims = input_blob->GetBlobDesc().dims;

    if (layer_param->exclusive_extend) {
        if (layer_param->axis < 0) {
            layer_param->axis += input_blob->GetBlobDesc().dims.size();
        }
        output_blob->GetBlobDesc().dims[layer_param->axis] = input_blob->GetBlobDesc().dims[layer_param->axis]+1;
    }

    return TNN_OK;
}

REGISTER_LAYER(Cumsum, LAYER_CUMSUM);

}  // namespace TNN_NS
