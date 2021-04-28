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
// specific language governing permissions and limitations under the License./

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CUDA_ACC(RoiAlign, LAYER_ROIALIGN);

__device__ float bilinear_interpolate(const float* input_data, int height, int width, float y,
        float x, bool is_mode_avg) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        // empty
        return 0;
    }

    if (y <= 0) {
        y = 0;
    }

    if (x <= 0) {
        x = 0;
    }

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (float)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (float)x_low;
    } else {
        x_high = x_low + 1;
    }

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    float v1 = input_data[y_low * width + x_low];
    float v2 = input_data[y_low * width + x_high];
    float v3 = input_data[y_high * width + x_low];
    float v4 = input_data[y_high * width + x_high];
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    float val = is_mode_avg
            ? (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)  // mode Avg
            : max(max(max(w1 * v1, w2 * v2), w3 * v3), w4 * v4);  // mode Max

    return val;
}

__global__ void roialign_kernel(int count, const float* input_data, float spatial_scale,
        int channels, int height, int width, int pooled_height, int pooled_width,
        int sampling_ratio, const float* input_rois, int roi_cols, float* output_data,
        bool is_mode_avg, const int* batch_indices_ptr) {
    CUDA_KERNEL_LOOP(index, count) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        // RoI could have 4 or 5 columns
        const float* offset_input_rois = input_rois + n * roi_cols;
        const auto roi_batch_ind = batch_indices_ptr[n];

        // Do not using rounding; this implementation detail is critical
        float roi_offset = 0.f;
        float roi_start_w = offset_input_rois[0] * spatial_scale - roi_offset;
        float roi_start_h = offset_input_rois[1] * spatial_scale - roi_offset;
        float roi_end_w = offset_input_rois[2] * spatial_scale - roi_offset;
        float roi_end_h = offset_input_rois[3] * spatial_scale - roi_offset;

        float roi_width = roi_end_w - roi_start_w;
        float roi_height = roi_end_h - roi_start_h;
        roi_width = max(roi_width, 1.f);
        roi_height = max(roi_height, 1.f);
        float bin_size_h = roi_height / pooled_height;
        float bin_size_w = roi_width / pooled_width;

        const float* offset_input_data = input_data +
            static_cast<int64_t>((roi_batch_ind * channels + c) * height * width);

        int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_height / pooled_height);
        int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_width / pooled_width);

        const float count = roi_bin_grid_h * roi_bin_grid_w;

        float output_val = 0.;
        bool max_flag = false;
        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            float y = roi_start_h + ph * bin_size_h +
                static_cast<float>(iy + .5f) * bin_size_h / static_cast<float>(roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                float x = roi_start_w + pw * bin_size_w +
                    static_cast<float>(ix + .5f) * bin_size_w / static_cast<float>(roi_bin_grid_w);

                float val = bilinear_interpolate(offset_input_data, height, width, y, x, is_mode_avg);

                if (is_mode_avg) {
                    output_val += val;
                } else {
                    if (!max_flag) {
                        output_val = val;
                        max_flag = true;
                    } else {
                        output_val = max(output_val, val);
                    }
                }
            }
        }
        if (is_mode_avg) {
            output_val /= count;
        }
        output_data[index] = output_val;
    }
}


Status CudaRoiAlignLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return CudaLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CudaRoiAlignLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CudaRoiAlignLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<RoiAlignLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    Blob *input_blob  = inputs[0];
    Blob *rois_blob = inputs[1];
    Blob *batch_indices_blob = inputs[2];
    Blob *output_blob = outputs[0];
    int count = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims);
    int channels = input_blob->GetBlobDesc().dims[1];
    int height = input_blob->GetBlobDesc().dims[2];
    int width = input_blob->GetBlobDesc().dims[3];
    int pooled_height = param->output_height;
    int pooled_width = param->output_width;
    float spatial_scale = param->spatial_scale;
    int sampling_ratio = param->sampling_ratio;
    int roi_cols = batch_indices_blob->GetBlobDesc().dims[0];
    float *input_data = static_cast<float *>(input_blob->GetHandle().base);
    float *input_rois = static_cast<float *>(rois_blob->GetHandle().base);
    int *batch_indices_ptr = static_cast<int *>(batch_indices_blob->GetHandle().base);
    float *output_data = static_cast<float *>(output_blob->GetHandle().base);
    roialign_kernel<<<TNN_CUDA_GET_BLOCKS(count), TNN_CUDA_NUM_THREADS, 0, context_->GetStream()>>>(count, input_data,
        spatial_scale, channels, height, width, pooled_height, pooled_width, sampling_ratio, input_rois, roi_cols,
        output_data, param->mode, batch_indices_ptr);
    
    return TNN_OK;
}

REGISTER_CUDA_ACC(RoiAlign, LAYER_ROIALIGN);

}  // namespace TNN_NS

