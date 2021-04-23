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

DECLARE_LAYER(RoiAlign, LAYER_ROIALIGN);

Status RoiAlignLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status RoiAlignLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    auto* input_blob        = input_blobs_[0];
    auto* rois              = input_blobs_[1];
    auto* output_blob       = output_blobs_[0];
    auto* param             = dynamic_cast<RoiAlignLayerParam*>(param_);
    const int num_rois      = rois->GetBlobDesc().dims[0];
    const int channels      = input_blob->GetBlobDesc().dims[1];
    const int output_height = param->output_height;
    const int output_width  = param->output_width;

    output_blob->GetBlobDesc().dims = {num_rois, channels, output_height, output_width};

    return TNN_OK;
}

REGISTER_LAYER(RoiAlign, LAYER_ROIALIGN);

}  // namespace TNN_NS
