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

DECLARE_LAYER(Reorg, LAYER_REORG);

Status ReorgLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status ReorgLayer::InferOutputShape() {
    Blob* input_blob = input_blobs_[0];

    ReorgLayerParam* reorg_param = dynamic_cast<ReorgLayerParam*>(param_);
    CHECK_PARAM_NULL(reorg_param);

    bool forward = reorg_param->forward;
    int stride   = reorg_param->stride;
    int mode     = reorg_param->mode;
    if (forward == false && mode == 1) {
        LOGE("Layer Reorg: do not support Reorg(SpaceToDepth) use CRD mode \n");
        return TNNERR_LAYER_ERR;
    }

    auto dims_input = input_blob->GetBlobDesc().dims;
    int num         = dims_input[0];
    int channels    = dims_input[1];
    int height      = dims_input[2];
    int width       = dims_input[3];
    int reorged_channels, reorged_height, reorged_width;

    if (forward) {
        if (channels % (stride * stride) != 0) {
            return Status(TNNERR_LAYER_ERR, "Error: channel and parameter stride is not compatible");
        }
        reorged_channels = channels / (stride * stride);
        reorged_width    = width * stride;
        reorged_height   = height * stride;
    } else {
        if ((height % stride != 0) || (width % stride != 0)) {
            return Status(TNNERR_LAYER_ERR, "Error: size and parameter stride is not compatible");
        }
        reorged_channels = channels * stride * stride;
        reorged_height   = height / stride;
        reorged_width    = width / stride;
    }

    DimsVector output_dims;
    output_dims.push_back(num);
    output_dims.push_back(reorged_channels);
    output_dims.push_back(reorged_height);
    output_dims.push_back(reorged_width);

    output_blobs_[0]->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(Reorg, LAYER_REORG);

}  // namespace TNN_NS
