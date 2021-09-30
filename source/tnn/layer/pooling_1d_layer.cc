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

DECLARE_LAYER(Pooling1D, LAYER_POOLING_1D);

inline int Pooling1DLayerRuntimeKernelHeight(PoolingLayerParam* pool_param, DimsVector input_dims) {
    int kernel_h = pool_param->kernels_params[0];
    // If the kernel_h and kernel_w are zero, it means the kernel size is
    // equal to the input height and width
    if (kernel_h == 0) {
        kernel_h = input_dims[2];  // NCDHW
    }

    if (pool_param->kernel_indexs[0] != -1) {
        kernel_h = input_dims[pool_param->kernel_indexs[0]];
    }
    pool_param->kernels[0] = kernel_h;
    return kernel_h;
}

Status Pooling1DLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status Pooling1DLayer::InferOutputShape(bool ignore_error) {
    BaseLayer::InferOutputShape(ignore_error);

    Blob* input_blob = input_blobs_[0];

    PoolingLayerParam* pool_param = dynamic_cast<PoolingLayerParam*>(param_);
    CHECK_PARAM_NULL(pool_param);

    auto dims_input = input_blob->GetBlobDesc().dims;
    int num         = input_blob->GetBlobDesc().dims[0];
    int channels    = input_blob->GetBlobDesc().dims[1];
    int height      = input_blob->GetBlobDesc().dims[2];

    const int kernel_h = Pooling1DLayerRuntimeKernelHeight(pool_param, dims_input);
    int stride_h       = pool_param->strides[0];

    // default padding following the proto setting
    int height_out = 0;
    if (pool_param->pad_type == -1) {
        int pad_h = pool_param->pads[0];
        if (pool_param->ceil_mode == 1) {
            height_out = int(std::ceil(float(height + 2 * pad_h - kernel_h) / (float)stride_h + 1));
        } else {
            height_out = int(std::floor(float(height + 2 * pad_h - kernel_h) / (float)stride_h + 1));
        }

        int pad_along_height = ((height_out - 1) * stride_h + kernel_h - height);
        int pad_down         = pad_along_height - pad_h;
        if (pad_down < 0) {
            pad_down = std::max(pad_down, 0);

            // verify
            int rectify_height_out = 0;
            if (pool_param->ceil_mode == 1) {
                rectify_height_out = int(std::ceil(float(height + pad_h + pad_down - kernel_h) / (float)stride_h + 1));
            } else {
                rectify_height_out = int(std::floor(float(height + pad_h + pad_down - kernel_h) / (float)stride_h + 1));
            }

            if (rectify_height_out != height_out) {
                LOGE_IF(!ignore_error, "Error: Pooling1DLayer, maybe it is the case for global pooling\n");
                return Status(TNNERR_PARAM_ERR, "Error: Pooling1DLayer, maybe it is the case for global pooling");
            }
        }
        pool_param->pads[1] = pad_down;
    } else {
        // The code below is based on the logic from
        // https://www.tensorflow.org/api_docs/python/nn/convolution
        if (pool_param->pad_type == 0)  // SAME type
        {
            if (pool_param->ceil_mode == 1) {
                height_out = int(std::ceil(float(height) / float(stride_h)));
            } else {
                height_out = int(std::floor(float(height) / float(stride_h)));
            }
        } else if (pool_param->pad_type == 1)  // VALID type
        {
            height_out = int(std::ceil(float(height - kernel_h + 1) / float(stride_h)));
        } else {
            LOGE_IF(!ignore_error, "Error: Pooling3DLayer, maybe it is the case for global pooling\n");
            return Status(TNNERR_PARAM_ERR, "Error: Pooling3DLayer, maybe it is the case for global pooling");
        }

        int pad_along_height = ((height_out - 1) * stride_h + kernel_h - height);

        // align with pytorch in ceil mode.
        int pad_top = int(std::ceil(float(pad_along_height) / float(stride_h)));

        int pad_down = pad_along_height - pad_top;

        pool_param->pads[0] = pad_top;
        pool_param->pads[1] = pad_down;
    }

    DimsVector output_dims;
    output_dims.push_back(num);
    output_dims.push_back(channels);
    output_dims.push_back(height_out);

    for (int i = 0; i < output_blobs_.size(); ++i) {
        output_blobs_[i]->GetBlobDesc().dims = output_dims;
    }
    return TNN_OK;
}

REGISTER_LAYER(Pooling1D, LAYER_POOLING_1D);

}  // namespace TNN_NS
