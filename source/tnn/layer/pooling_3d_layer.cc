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

DECLARE_LAYER(Pooling3D, LAYER_POOLING_3D);

inline int Pooling3DLayerRuntimeKernelHeight(PoolingLayerParam* pool_param, DimsVector input_dims) {
    int kernel_h = pool_param->kernels_params[1];
    // If the kernel_h and kernel_w are zero, it means the kernel size is
    // equal to the input height and width
    if (kernel_h == 0) {
        kernel_h = input_dims[3];  // NCDHW
    }

    if (pool_param->kernel_indexs[1] != -1) {
        kernel_h = input_dims[pool_param->kernel_indexs[1]];
    }
    pool_param->kernels[1] = kernel_h;
    return kernel_h;
}

inline int Pooling3DLayerRuntimeKernelWidth(PoolingLayerParam* pool_param, DimsVector input_dims) {
    int kernel_w = pool_param->kernels_params[0];
    // If the kernel_h and kernel_w are zero, it means the kernel size is
    // equal to the input height and width
    if (kernel_w == 0) {
        kernel_w = input_dims[4];  // NCDHW
    }

    if (pool_param->kernel_indexs[0] != -1) {
        kernel_w = input_dims[pool_param->kernel_indexs[0]];
    }
    pool_param->kernels[0] = kernel_w;
    return kernel_w;
}

inline int Pooling3DLayerRuntimeKernelDepth(PoolingLayerParam* pool_param, DimsVector input_dims) {
    int kernel_d = pool_param->kernels_params[2];
    // If the kernel_h and kernel_w are zero, it means the kernel size is
    // equal to the input height and width
    if (kernel_d == 0) {
        kernel_d = input_dims[2];  // NCDHW
    }

    if (pool_param->kernel_indexs[2] != -1) {
        kernel_d = input_dims[pool_param->kernel_indexs[2]];
    }
    pool_param->kernels[2] = kernel_d;
    return kernel_d;
}

Status Pooling3DLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status Pooling3DLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);
    
    Blob* input_blob = input_blobs_[0];

    PoolingLayerParam* pool_param = dynamic_cast<PoolingLayerParam*>(param_);
    CHECK_PARAM_NULL(pool_param);

    auto dims_input = input_blob->GetBlobDesc().dims;
    int num         = input_blob->GetBlobDesc().dims[0];
    int channels    = input_blob->GetBlobDesc().dims[1];
    int depth       = input_blob->GetBlobDesc().dims[2];
    int height      = input_blob->GetBlobDesc().dims[3];
    int width       = input_blob->GetBlobDesc().dims[4];

    const int kernel_w = Pooling3DLayerRuntimeKernelWidth(pool_param, dims_input);
    const int kernel_h = Pooling3DLayerRuntimeKernelHeight(pool_param, dims_input);
    const int kernel_d = Pooling3DLayerRuntimeKernelDepth(pool_param, dims_input);

    int stride_w = pool_param->strides[0];
    int stride_h = pool_param->strides[1];
    int stride_d = pool_param->strides[2];

    int height_out = 0;
    int width_out  = 0;
    int depth_out  = 0;

    // default padding following the proto setting
    if (pool_param->pad_type == -1) {
        int pad_w = pool_param->pads[0];
        int pad_h = pool_param->pads[2];
        int pad_d = pool_param->pads[4];
        if (pool_param->ceil_mode == 1) {
            height_out = int(std::ceil(float(height + 2 * pad_h - kernel_h) / (float)stride_h + 1));
            width_out  = int(std::ceil(float(width + 2 * pad_w - kernel_w) / (float)stride_w + 1));
            depth_out  = int(std::ceil(float(depth + 2 * pad_d - kernel_d) / (float)stride_d + 1));
        } else {
            height_out = int(std::floor(float(height + 2 * pad_h - kernel_h) / (float)stride_h + 1));
            width_out  = int(std::floor(float(width + 2 * pad_w - kernel_w) / (float)stride_w + 1));
            depth_out  = int(std::floor(float(depth + 2 * pad_d - kernel_d) / (float)stride_d + 1));
        }

        int pad_along_height = ((height_out - 1) * stride_h + kernel_h - height);
        int pad_along_width  = ((width_out - 1) * stride_w + kernel_w - width);
        int pad_along_depth  = ((depth_out - 1) * stride_d + kernel_d - depth);
        int pad_down         = pad_along_height - pad_h;
        int pad_right        = pad_along_width - pad_w;
        int pad_back         = pad_along_depth - pad_d;
        if (pad_down < 0 || pad_right < 0 || pad_back < 0) {
            pad_down  = std::max(pad_down, 0);
            pad_right = std::max(pad_right, 0);
            pad_back  = std::max(pad_back, 0);

            // verify
            int rectify_height_out = 0;
            int rectify_width_out  = 0;
            int rectify_depth_out  = 0;
            if (pool_param->ceil_mode == 1) {
                rectify_height_out = int(std::ceil(float(height + pad_h + pad_down - kernel_h) / (float)stride_h + 1));
                rectify_width_out  = int(std::ceil(float(width + pad_w + pad_right - kernel_w) / (float)stride_w + 1));
                rectify_depth_out  = int(std::ceil(float(depth + pad_d + pad_back  - kernel_d) / (float)stride_d + 1));
            } else {
                rectify_height_out = int(std::floor(float(height + pad_h + pad_down - kernel_h) / (float)stride_h + 1));
                rectify_width_out  = int(std::floor(float(width + pad_w + pad_right - kernel_w) / (float)stride_w + 1));
                rectify_depth_out  = int(std::floor(float(depth + pad_d + pad_back  - kernel_d) / (float)stride_d + 1));
            }

            if (rectify_height_out != height_out || rectify_width_out != width_out || rectify_depth_out != depth_out) {
                LOGE_IF(!ignore_error, "Error: Pooling3DLayer, maybe it is the case for global pooling\n");
                return Status(TNNERR_PARAM_ERR, "Error: Pooling3DLayer, maybe it is the case for global pooling");
            }
        }

        pool_param->pads[1] = pad_right;
        pool_param->pads[3] = pad_down;
        pool_param->pads[5] = pad_back;
    } else {
        // The code below is based on the logic from
        // https://www.tensorflow.org/api_docs/python/nn/convolution
        if (pool_param->pad_type == 0)  // SAME type
        {
            if (pool_param->ceil_mode == 1) {
                height_out = int(std::ceil(float(height) / float(stride_h)));
                width_out  = int(std::ceil(float(width) / float(stride_w)));
                depth_out  = int(std::ceil(float(depth) / float(stride_d)));
            } else {
                height_out = int(std::floor(float(height) / float(stride_h)));
                width_out  = int(std::floor(float(width) / float(stride_w)));
                depth_out  = int(std::floor(float(depth) / float(stride_d)));
            }
        } else if (pool_param->pad_type == 1)  // VALID type
        {
            height_out = int(std::ceil(float(height - kernel_h + 1) / float(stride_h)));
            width_out  = int(std::ceil(float(width - kernel_w + 1) / float(stride_w)));
            depth_out  = int(std::ceil(float(depth - kernel_d + 1) / float(stride_d)));
        } else {
            LOGE_IF(!ignore_error, "Error: Pooling3DLayer, maybe it is the case for global pooling\n");
            return Status(TNNERR_PARAM_ERR, "Error: Pooling3DLayer, maybe it is the case for global pooling");
        }

        int pad_along_height = ((height_out - 1) * stride_h + kernel_h - height);
        int pad_along_width  = ((width_out - 1) * stride_w + kernel_w - width);
        int pad_along_depth  = ((depth_out - 1) * stride_d + kernel_d - depth);

        // align with pytorch in ceil mode.
        int pad_top   = int(std::ceil(float(pad_along_height) / float(stride_h)));
        int pad_left  = int(std::ceil(float(pad_along_width) / float(stride_h)));
        int pad_front = int(std::ceil(float(pad_along_depth) / float(stride_h)));

        int pad_down  = pad_along_height - pad_top;
        int pad_right = pad_along_width - pad_left;
        int pad_back  = pad_along_depth - pad_front;

        pool_param->pads[0] = pad_left;
        pool_param->pads[1] = pad_right;
        pool_param->pads[2] = pad_top;
        pool_param->pads[3] = pad_down;
        pool_param->pads[4] = pad_front;
        pool_param->pads[5] = pad_back;
    }

    DimsVector output_dims;
    output_dims.push_back(num);
    output_dims.push_back(channels);
    output_dims.push_back(depth_out);
    output_dims.push_back(height_out);
    output_dims.push_back(width_out);

    for (int i = 0; i < output_blobs_.size(); ++i) {
        output_blobs_[i]->GetBlobDesc().dims = output_dims;
    }
    return TNN_OK;
}

REGISTER_LAYER(Pooling3D, LAYER_POOLING_3D);

}  // namespace TNN_NS
