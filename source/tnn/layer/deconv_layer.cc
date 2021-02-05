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

#include <algorithm>
#include <cmath>

#include "tnn/layer/base_layer.h"

namespace TNN_NS {

DECLARE_LAYER(Deconv, LAYER_DECONVOLUTION);

Status DeconvLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status DeconvLayer::InferOutputShape() {
    Blob* input_blob             = input_blobs_[0];
    Blob* output_blob            = output_blobs_[0];
    ConvLayerParam* deconv_param = dynamic_cast<ConvLayerParam*>(param_);
    CHECK_PARAM_NULL(deconv_param);

    int num    = input_blob->GetBlobDesc().dims[0];
    int height = input_blob->GetBlobDesc().dims[2];
    int width  = input_blob->GetBlobDesc().dims[3];

    const int pad_w_begin = deconv_param->pads[0];
    const int pad_h_begin = deconv_param->pads[2];

    const int kernel_w = deconv_param->kernels[0];
    const int kernel_h = deconv_param->kernels[1];

    const int stride_w = deconv_param->strides[0];
    const int stride_h = deconv_param->strides[1];

    const int dilation_w = deconv_param->dialations[0];
    const int dilation_h = deconv_param->dialations[1];

    int height_out = 0;
    int width_out  = 0;

    const int pad_type = deconv_param->pad_type;

    int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    // Refactored the code to support tensorflow models
    if (pad_type == -1)  // default padding following the proto setting
    {
        height_out = stride_h * (height - 1) + kernel_extent_h - 2 * pad_h_begin;
        width_out  = stride_w * (width - 1) + kernel_extent_w - 2 * pad_w_begin;
    } else if (pad_type == 0 || pad_type == 1 || pad_type == 2 || pad_type == 3) {
        // The code below is based on the logic from tensorflow
        height_out = height * stride_h;
        width_out  = width * stride_w;
        if (pad_type == 0 || pad_type == 3)  // SAME type
        {
            height_out = height * stride_h;
            width_out  = width * stride_w;
        } else if (pad_type == 1)  // VALID type
        {
            height_out = height * stride_h + std::max(kernel_extent_h - stride_h, 0);
            width_out  = width * stride_w + std::max(kernel_extent_w - stride_w, 0);
        } else if (pad_type == 2)  // FULL type
        {
            height_out = height * stride_h - (stride_h + kernel_extent_h - 2);
            width_out  = width * stride_w - (stride_w + kernel_extent_w - 2);
        } else {
            LOGE("Error: DeconvLayer dont support pad type: %d\n", pad_type);
            return Status(TNNERR_PARAM_ERR, "Error: DeconvLayer dont support pad type");
        }

        int pad_along_height = ((height - 1) * stride_h + kernel_extent_h - height_out);
        int pad_along_width  = ((width - 1) * stride_w + kernel_extent_w - width_out);
        if (pad_type == 3) {
            pad_along_height = std::max(pad_along_height, 0);
            pad_along_width  = std::max(pad_along_width, 0);
        }

        int pad_top  = pad_along_height / 2;
        int pad_left = pad_along_width / 2;

        int pad_down  = pad_along_height - pad_top;
        int pad_right = pad_along_width - pad_left;

        // reset pad_h and pad_w
        deconv_param->pads[0] = pad_left;
        deconv_param->pads[1] = pad_right;
        deconv_param->pads[2] = pad_top;
        deconv_param->pads[3] = pad_down;

        if (pad_type == 3) {
            // deconv exchange pad_right and pad_left because of output_padding
            deconv_param->pads[0] = pad_right;
            deconv_param->pads[1] = pad_left;
            deconv_param->pads[2] = pad_down;
            deconv_param->pads[3] = pad_top;
        }
        //        LOGE("DeconvLayerpads: %d %d %d %d\n", deconv_param->pads[0],
        //             deconv_param->pads[1],
        //             deconv_param->pads[2],
        //             deconv_param->pads[3]);
    } else {
        LOGE("Error: DeconvLayer dont support pad type: %d\n", pad_type);
        return Status(TNNERR_PARAM_ERR, "Error: DeconvLayer dont support pad type");
    }

    int group = deconv_param->group;
    if (group == 0) {
        return Status(TNNERR_INVALID_GROUP, "Error: invalid group param");
    }

    if (height_out <= 0 || width_out <= 0) {
        LOGE("Error: invalid deconv param, height_out(%d) or width_out(%d) is less than zero\n", height_out, width_out);
        return Status(TNNERR_PARAM_ERR, "Error: invalid deconv param, height_out or width_out is less than zero");
    }

    DimsVector output_dims;
    output_dims.push_back(num);
    output_dims.push_back(deconv_param->output_channel);
    output_dims.push_back(height_out);
    output_dims.push_back(width_out);
    output_blob->GetBlobDesc().dims = output_dims;

    return TNN_OK;
}

REGISTER_LAYER(Deconv, LAYER_DECONVOLUTION);

}  // namespace TNN_NS
