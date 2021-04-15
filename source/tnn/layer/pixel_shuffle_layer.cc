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

namespace TNN_NS {

DECLARE_LAYER(PixelShuffle, LAYER_PIXEL_SHUFFLE);

Status PixelShuffleLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status PixelShuffleLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    auto input_blob        = input_blobs_[0];
    auto input_dims        = input_blob->GetBlobDesc().dims;
    auto output_blob       = output_blobs_[0];
    auto layer_param       = dynamic_cast<PixelShuffleLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);
    int upscale_factor     = layer_param->upscale_factor;
    DimsVector output_dims = input_dims;
    ASSERT(input_dims[1] >= upscale_factor && input_dims[1] % (upscale_factor * upscale_factor) == 0);
    output_dims[1]                  = input_dims[1] / (upscale_factor * upscale_factor);
    output_dims[2]                  = input_dims[2] * upscale_factor;
    output_dims[3]                  = input_dims[3] * upscale_factor;
    output_blob->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(PixelShuffle, LAYER_PIXEL_SHUFFLE);

}  // namespace TNN_NS
