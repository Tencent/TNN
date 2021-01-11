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

namespace TNN_NS {

DECLARE_LAYER(Split, LAYER_SPLITING);

Status SplitLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status SplitLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    Blob* input_blob = input_blobs_[0];

    for (size_t i = 0; i < output_blobs_.size(); i++) {
        output_blobs_[i]->GetBlobDesc().dims = input_blob->GetBlobDesc().dims;
    }

    return TNN_OK;
}

REGISTER_LAYER(Split, LAYER_SPLITING);

}  // namespace TNN_NS
