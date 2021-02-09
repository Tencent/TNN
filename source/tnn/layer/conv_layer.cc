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

DECLARE_LAYER(Conv, LAYER_CONVOLUTION);

Status ConvLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status ConvLayer::InferOutputShape() {
    Blob* input_blob           = input_blobs_[0];
    Blob* output_blob          = output_blobs_[0];
    ConvLayerParam* conv_param = dynamic_cast<ConvLayerParam*>(param_);
    CHECK_PARAM_NULL(conv_param);

    int num    = input_blob->GetBlobDesc().dims[0];
    int height = input_blob->GetBlobDesc().dims[2];
    int width  = input_blob->GetBlobDesc().dims[3];

    const int pad_w_begin = conv_param->pads[0];
    const int pad_h_begin = conv_param->pads[2];

    const int kernel_w = conv_param->kernels[0];
    const int kernel_h = conv_param->kernels[1];

    const int stride_w = conv_param->strides[0];
    const int stride_h = conv_param->strides[1];

    const int dilation_w = conv_param->dialations[0];
    const int dilation_h = conv_param->dialations[1];

    int height_out = 0;
    int width_out  = 0;

    const int pad_type = conv_param->pad_type;

    int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    // Refactored the code to support tensorflow models
    if (pad_type == -1)  // default padding following the proto setting
    {
        const int pad_left   = conv_param->pads[0];
        const int pad_right  = conv_param->pads[1];
        const int pad_top    = conv_param->pads[2];
        const int pad_bottom = conv_param->pads[3];

        height_out = (height + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1;
        width_out  = (width + pad_left + pad_right - kernel_extent_w) / stride_w + 1;

    } else if (pad_type == 0 || pad_type == 1 || pad_type == 2) {
        // The code below is based on the logic from
        // https://www.tensorflow.org/api_docs/python/nn/convolution
        if (pad_type == 0)  // SAME type
        {
            height_out = static_cast<int>(std::ceil(float(height) / float(stride_h)));
            width_out  = static_cast<int>(std::ceil(float(width) / float(stride_w)));
        } else if (pad_type == 1)  // VALID type
        {
            height_out = static_cast<int>(std::ceil(float(height - kernel_extent_h + 1) / float(stride_h)));
            width_out  = static_cast<int>(std::ceil(float(width - kernel_extent_w + 1) / float(stride_w)));
        } else  // FULL type
        {
            // to-do: deconv has full type, what's conv's full type?
            LOGE("Error: ConvLayer dont support pad type: %d\n", pad_type);
            return Status(TNNERR_PARAM_ERR, "Error: ConvLayer dont support pad type");
        }

        int pad_along_height = ((height_out - 1) * stride_h + kernel_extent_h - height);
        int pad_along_width  = ((width_out - 1) * stride_w + kernel_extent_w - width);
        int pad_top          = pad_along_height / 2;
        int pad_left         = pad_along_width / 2;

        int pad_down  = pad_along_height - pad_top;
        int pad_right = pad_along_width - pad_left;
        if (pad_down < 0) {
            pad_down = 0;
        }
        if (pad_right < 0) {
            pad_right = 0;
        }

        // reset pad_h and pad_w
        conv_param->pads[0] = pad_left;
        conv_param->pads[1] = pad_right;
        conv_param->pads[2] = pad_top;
        conv_param->pads[3] = pad_down;
    } else {
        LOGE("Error: ConvLayer dont support pad type: %d\n", pad_type);
        return Status(TNNERR_PARAM_ERR, "Error: ConvLayer dont support pad type");
    }

    int group = conv_param->group;
    if (group == 0) {
        LOGE("Error: ConvLayer Error: invalid group param\n");
        return Status(TNNERR_INVALID_GROUP, "ConvLayer Error: invalid group param");
    }

    if (height_out <= 0 || width_out <= 0) {
        LOGE("Error: invalid deconv param, height_out(%d) or width_out(%d) is less than zero\n", height_out, width_out);
        return Status(TNNERR_PARAM_ERR, "invalid conv param, height_out or width_out is less than zero");
    }

    DimsVector output_dims;
    output_dims.push_back(num);
    output_dims.push_back(conv_param->output_channel);
    output_dims.push_back(height_out);
    output_dims.push_back(width_out);
    output_blob->GetBlobDesc().dims = output_dims;

    return TNN_OK;
}

REGISTER_LAYER(Conv, LAYER_CONVOLUTION);

}  // namespace TNN_NS
