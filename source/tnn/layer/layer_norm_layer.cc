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

namespace TNN_NS {

DECLARE_LAYER(LayerNorm, LAYER_LAYER_NORM);

Status LayerNormLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status LayerNormLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    if (input_blobs_.size() < 3) {
        return Status(TNNERR_PARAM_ERR, "LayerNormLayer has no input blob of scale or bias");
    }
    
    output_blobs_[0]->GetBlobDesc().dims = input_blobs_[0]->GetBlobDesc().dims;
    
    return TNN_OK;
}

REGISTER_LAYER(LayerNorm, LAYER_LAYER_NORM);

}  // namespace TNN_NS
