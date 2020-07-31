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

DECLARE_LAYER(RoiAlign, LAYER_ROI_ALIGN);

Status RoiAlignLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status RoiAlignLayer::InferOutputShape() {
    auto input_blob     = input_blobs_[0];
    auto num_rois       = input_blobs_[1]->GetBlobDesc().dims[0];

    auto roi_align_param = dynamic_cast<RoiAlignLayerParam*>(param_);

    ASSERT(input_blob->GetBlobDesc().dims.size() == 4);

    DimsVector output_dims;
    output_dims.push_back(num_rois);
    output_dims.push_back(input_blob->GetBlobDesc().dims[1]);
    output_dims.push_back(roi_align_param->output_height);
    output_dims.push_back(roi_align_param->output_width);

    for (int i = 0; i < output_blobs_.size(); ++i) {
        output_blobs_[i]->GetBlobDesc().dims = output_dims;
    }

    return TNN_OK;
}

REGISTER_LAYER(RoiAlign, LAYER_ROI_ALIGN);

}  // namespace TNN_NS