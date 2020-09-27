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

DECLARE_LAYER(Upsample, LAYER_UPSAMPLE);

Status UpsampleLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status UpsampleLayer::InferOutputShape() {
    Blob* input_blob = input_blobs_[0];

    UpsampleLayerParam* layer_param =
        dynamic_cast<UpsampleLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);

    int num      = input_blob->GetBlobDesc().dims[0];
    int channels = input_blob->GetBlobDesc().dims[1];
    int height   = input_blob->GetBlobDesc().dims[2];
    int width    = input_blob->GetBlobDesc().dims[3];
    int width_out  = 0;
    int height_out = 0;

    if (layer_param->mode == 1 || layer_param->mode == 2) {
        //floor is wrong for some model
        width_out  = int(round(width * layer_param->scales[0]));
        height_out = int(round(height * layer_param->scales[1]));
    } else {
        LOGE("Error: unsupport upsample type:%d", layer_param->mode);
        return Status(TNNERR_PARAM_ERR, "unsupport upsample type");
    }

    if (layer_param->dims.size() >= 2) {
        width_out  = (int)layer_param->dims[0];
        height_out = (int)layer_param->dims[1];
    }

    if (width_out <=0 || height_out <=0) {
        LOGE("Error: UpsampleLayer invalid output shape: height(%d) width(%d)", height_out, width_out);
        return Status(TNNERR_PARAM_ERR, "UpsampleLayer invalid output shape");
    }

    DimsVector output_dims;
    output_dims.push_back(num);
    output_dims.push_back(channels);
    output_dims.push_back(height_out);
    output_dims.push_back(width_out);

    output_blobs_[0]->GetBlobDesc().dims = output_dims;
    return TNN_OK;
}

REGISTER_LAYER(Upsample, LAYER_UPSAMPLE);

}  // namespace TNN_NS
