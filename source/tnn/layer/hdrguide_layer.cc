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
DECLARE_LAYER(HdrGuide, LAYER_HDRGUIDE);

Status HdrGuideLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status HdrGuideLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    Blob* input_blob  = input_blobs_[0];
    Blob* output_blob = output_blobs_[0];

    int num     = input_blob->GetBlobDesc().dims[0];
    int channel = input_blob->GetBlobDesc().dims[1];
    int height  = input_blob->GetBlobDesc().dims[2];
    int width   = input_blob->GetBlobDesc().dims[3];

    if (channel != 3) {
        LOGE_IF(!ignore_error,
            "Error: HdrGuideLayer Error: invalid channel size (need to be "
            "3)\n");
        return Status(TNNERR_PARAM_ERR, "HdrGuideLayer Error: invalid channel size");
    }

    if (height <= 0 || width <= 0) {
        LOGE_IF(!ignore_error, "Error: invalid height or width, is less than zero\n");
        return Status(TNNERR_PARAM_ERR, "invalid height or width, is less than zero");
    }

    DimsVector output_dims;
    output_dims.push_back(num);
    output_dims.push_back(1);
    output_dims.push_back(height);
    output_dims.push_back(width);
    output_blob->GetBlobDesc().dims = output_dims;

    return TNN_OK;
}

REGISTER_LAYER(HdrGuide, LAYER_HDRGUIDE);

}  // namespace TNN_NS
