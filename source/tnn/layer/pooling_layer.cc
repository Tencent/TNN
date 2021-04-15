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

DECLARE_LAYER(Pooling, LAYER_POOLING);

inline int PoolingLayerRuntimeKernelHeight(PoolingLayerParam* pool_param, DimsVector input_dims) {
    int kernel_h = pool_param->kernels_params[1];
    // If the kernel_h and kernel_w are zero, it means the kernel size is
    // equal to the input height and width
    if (kernel_h == 0) {
        kernel_h = input_dims[2];
    }

    if (pool_param->kernel_indexs[1] != -1) {
        kernel_h = input_dims[pool_param->kernel_indexs[1]];
    }
    pool_param->kernels[1] = kernel_h;
    return kernel_h;
}

inline int PoolingLayerRuntimeKernelWidth(PoolingLayerParam* pool_param, DimsVector input_dims) {
    int kernel_w = pool_param->kernels_params[0];
    // If the kernel_h and kernel_w are zero, it means the kernel size is
    // equal to the input height and width
    if (kernel_w == 0) {
        kernel_w = input_dims[3];
    }

    if (pool_param->kernel_indexs[0] != -1) {
        kernel_w = input_dims[pool_param->kernel_indexs[0]];
    }
    pool_param->kernels[0] = kernel_w;
    return kernel_w;
}

Status PoolingLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status PoolingLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    Blob* input_blob = input_blobs_[0];

    PoolingLayerParam* pool_param = dynamic_cast<PoolingLayerParam*>(param_);
    CHECK_PARAM_NULL(pool_param);

    auto dims_input = input_blob->GetBlobDesc().dims;
    int num         = dims_input[0];
    int channels    = dims_input[1];
    int height      = dims_input[2];
    int width       = dims_input[3];

    if (pool_param->is_adaptive_pool) {
        const int output_blobs_size = output_blobs_.size();
        const auto output_shape     = pool_param->output_shape;
        for (int i = 0; i < output_blobs_size; i++) {
            output_blobs_[i]->GetBlobDesc().dims = {num, channels, output_shape[1], output_shape[0]};
        }

        return TNN_OK;
    }

    const int kernel_w = PoolingLayerRuntimeKernelWidth(pool_param, dims_input);
    const int kernel_h = PoolingLayerRuntimeKernelHeight(pool_param, dims_input);
    int stride_w       = pool_param->strides[0];
    int stride_h       = pool_param->strides[1];

    // int height_out = (int)ceil((height + 2 * pad_h - kernel_h) /
    // (float)stride_h) + 1; int width_out = (int)ceil((width + 2 * pad_w -
    // kernel_w) / (float)stride_w) + 1;

    int height_out = 0;
    int width_out  = 0;

    if (pool_param->pad_type == -1)  // default padding following the proto setting
    {
        int pad_left  = pool_param->pads[0];
        int pad_right = pool_param->pads[1];
        int pad_top   = pool_param->pads[2];
        int pad_down  = pool_param->pads[3];
        if (pool_param->ceil_mode == 1) {
            height_out =
                static_cast<int>(std::ceil(float(height + pad_top + pad_down - kernel_h) / (float)stride_h + 1));
            width_out =
                static_cast<int>(std::ceil(float(width + pad_left + pad_right - kernel_w) / (float)stride_w + 1));
        } else {
            height_out =
                static_cast<int>(std::floor(float(height + pad_top + pad_down - kernel_h) / (float)stride_h + 1));
            width_out =
                static_cast<int>(std::floor(float(width + pad_left + pad_right - kernel_w) / (float)stride_w + 1));
        }

        int pad_along_height = ((height_out - 1) * stride_h + kernel_h - height);
        int pad_along_width  = ((width_out - 1) * stride_w + kernel_w - width);
        pad_down             = pad_along_height - pad_top;
        pad_right            = pad_along_width - pad_left;
        if (pad_down < 0 || pad_right < 0) {
            pad_down  = std::max(pad_down, 0);
            pad_right = std::max(pad_right, 0);

            // verify
            int rectify_height_out = 0;
            int rectify_width_out  = 0;
            if (pool_param->ceil_mode == 1) {
                rectify_height_out =
                    static_cast<int>(std::ceil(float(height + pad_top + pad_down - kernel_h) / (float)stride_h + 1));
                rectify_width_out =
                    static_cast<int>(std::ceil(float(width + pad_left + pad_right - kernel_w) / (float)stride_w + 1));
            } else {
                rectify_height_out =
                    static_cast<int>(std::floor(float(height + pad_top + pad_down - kernel_h) / (float)stride_h + 1));
                rectify_width_out =
                    static_cast<int>(std::floor(float(width + pad_down + pad_right - kernel_w) / (float)stride_w + 1));
            }

            if (rectify_height_out != height_out || rectify_width_out != width_out) {
                LOGE_IF(!ignore_error, "Error: PoolingLayer, maybe it is the case for global pooling\n");
                return Status(TNNERR_PARAM_ERR, "Error: Pooling3DLayer, maybe it is the case for global pooling");
            }
        }

        pool_param->pads[1] = pad_right;
        pool_param->pads[3] = pad_down;
    } else {
        // The code below is based on the logic from
        // https://www.tensorflow.org/api_docs/python/nn/convolution
        if (pool_param->pad_type == 0)  // SAME type
        {
            height_out = static_cast<int>(std::ceil(float(height) / float(stride_h)));
            width_out  = static_cast<int>(std::ceil(float(width) / float(stride_w)));
        } else if (pool_param->pad_type == 1)  // VALID type
        {
            height_out = static_cast<int>(std::ceil(float(height - kernel_h + 1) / float(stride_h)));
            width_out  = static_cast<int>(std::ceil(float(width - kernel_w + 1) / float(stride_w)));
        } else {
            LOGE_IF(!ignore_error, "Error: PoolingLayer %s, maybe it is the case for global pooling\n",
                    GetLayerName().c_str());
            return Status(TNNERR_PARAM_ERR, "Error: PoolingLayer, maybe it is the case for global pooling");
        }

        int pad_along_height = ((height_out - 1) * stride_h + kernel_h - height);
        int pad_along_width  = ((width_out - 1) * stride_w + kernel_w - width);

        int pad_top  = pad_along_height / 2;
        int pad_left = pad_along_width / 2;

        int pad_down  = pad_along_height - pad_top;
        int pad_right = pad_along_width - pad_left;

        pool_param->pads[0] = pad_left;
        pool_param->pads[1] = pad_right;
        pool_param->pads[2] = pad_top;
        pool_param->pads[3] = pad_down;
    }

    DimsVector output_dims;
    output_dims.push_back(num);
    output_dims.push_back(channels);
    output_dims.push_back(height_out);
    output_dims.push_back(width_out);

    for (int i = 0; i < output_blobs_.size(); ++i) {
        output_blobs_[i]->GetBlobDesc().dims = output_dims;
    }
    return TNN_OK;
}

REGISTER_LAYER(Pooling, LAYER_POOLING);

}  // namespace TNN_NS
