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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
DECLARE_LAYER(Unsqueeze, LAYER_UNSQUEEZE);

Status UnsqueezeLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status UnsqueezeLayer::InferOutputShape() {
    ASSERT(input_blobs_.size() == 1);
    const auto& input_blob  = input_blobs_[0];
    const auto& input_dims  = input_blob->GetBlobDesc().dims;
    const auto& output_blob = output_blobs_[0];
    auto& output_dims       = output_blob->GetBlobDesc().dims;
    // the output blob has only one dim, the value is the size of input blob dims
    auto layer_param      = dynamic_cast<UnsqueezeLayerParam*>(param_);
    auto axes             = layer_param->axes;
    auto output_dims_size = axes.size() + input_dims.size();
    output_dims = input_dims;
    for (const auto& axis : axes) {
        output_dims.insert(output_dims.begin() + axis, 1);
    }
    return TNN_OK;
}

REGISTER_LAYER(Unsqueeze, LAYER_UNSQUEEZE);

}  // namespace TNN_NS