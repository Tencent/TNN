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

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

struct PreCalc {
    int pos1;
    int pos2;
    int pos3;
    int pos4;
    float w1;
    float w2;
    float w3;
    float w4;
};

void pre_calc_for_bilinear_interpolate(const int height, const int width, const int pooled_height,
                                       const int pooled_width, const int iy_upper, const int ix_upper,
                                       float roi_start_h, float roi_start_w, float bin_size_h, float bin_size_w,
                                       int roi_bin_grid_h, int roi_bin_grid_w, std::vector<PreCalc> &pre_calc) {
    int pre_calc_index = 0;
    for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
            for (int iy = 0; iy < iy_upper; iy++) {
                const float yy =
                    roi_start_h + ph * bin_size_h +
                    static_cast<float>(iy + .5f) * bin_size_h / static_cast<float>(roi_bin_grid_h);  // e.g., 0.5, 1.5
                for (int ix = 0; ix < ix_upper; ix++) {
                    const float xx = roi_start_w + pw * bin_size_w +
                                     static_cast<float>(ix + .5f) * bin_size_w / static_cast<float>(roi_bin_grid_w);

                    float x = xx;
                    float y = yy;
                    // deal with: inverse elements are out of feature map boundary
                    if (y < -1.0 || y > height || x < -1.0 || x > width) {
                        // empty
                        PreCalc pc;
                        pc.pos1                  = 0;
                        pc.pos2                  = 0;
                        pc.pos3                  = 0;
                        pc.pos4                  = 0;
                        pc.w1                    = 0;
                        pc.w2                    = 0;
                        pc.w3                    = 0;
                        pc.w4                    = 0;
                        pre_calc[pre_calc_index] = pc;
                        pre_calc_index += 1;
                        continue;
                    }

                    if (y <= 0) {
                        y = 0;
                    }
                    if (x <= 0) {
                        x = 0;
                    }

                    auto y_low = static_cast<int>(y);
                    auto x_low = static_cast<int>(x);
                    int y_high;
                    int x_high;

                    if (y_low >= height - 1) {
                        y_high = y_low = height - 1;
                        y              = (float)y_low;
                    } else {
                        y_high = y_low + 1;
                    }

                    if (x_low >= width - 1) {
                        x_high = x_low = width - 1;
                        x              = (float)x_low;
                    } else {
                        x_high = x_low + 1;
                    }

                    auto ly = y - y_low;
                    auto lx = x - x_low;
                    auto hy = static_cast<float>(1.) - ly;
                    auto hx = static_cast<float>(1.) - lx;
                    auto w1 = hy * hx;
                    auto w2 = hy * lx;
                    auto w3 = ly * hx;
                    auto w4 = ly * lx;

                    // save weights and indeces
                    PreCalc pc;
                    pc.pos1                  = y_low * width + x_low;
                    pc.pos2                  = y_low * width + x_high;
                    pc.pos3                  = y_high * width + x_low;
                    pc.pos4                  = y_high * width + x_high;
                    pc.w1                    = w1;
                    pc.w2                    = w2;
                    pc.w3                    = w3;
                    pc.w4                    = w4;
                    pre_calc[pre_calc_index] = pc;

                    pre_calc_index += 1;
                }
            }
        }
    }
}

DECLARE_CPU_ACC(RoiAlign, LAYER_ROI_ALIGN);

Status CpuRoiAlignLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuRoiAlignLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<RoiAlignLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: RoiAlignLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: RoiAlignLayerParam is nil");
    }

    Blob *input_blob       = inputs[0];
    Blob *rois             = inputs[1];
    Blob *output_blob      = outputs[0];
    const int channel_size = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, 2);
    if (0 == channel_size) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    auto n_rois         = rois->GetBlobDesc().dims[1];
    auto num_roi_cols   = rois->GetBlobDesc().dims[2];
    auto channels       = input_blob->GetBlobDesc().dims[1];
    auto height         = input_blob->GetBlobDesc().dims[2];
    auto width          = input_blob->GetBlobDesc().dims[3];
    auto sampling_ratio = layer_param->sampling_ratio;
    auto spatial_scale  = layer_param->spatial_scale;
    auto pooled_height  = layer_param->output_height;
    auto pooled_width   = layer_param->output_width;

    float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data = static_cast<float *>(output_blob->GetHandle().base);
    float *rois_data   = static_cast<float *>(rois->GetHandle().base);

    for (int n = 0; n < n_rois; ++n) {
        auto index_n = n * channels * pooled_width * pooled_height;

        const float *offset_bottom_rois = rois_data + n * num_roi_cols;
        const auto roi_batch_ind        = offset_bottom_rois[0];

        // Do not using rounding; this implementation detail is critical
        auto roi_start_w = offset_bottom_rois[1] * spatial_scale;
        auto roi_start_h = offset_bottom_rois[2] * spatial_scale;
        auto roi_end_w   = offset_bottom_rois[3] * spatial_scale;
        auto roi_end_h   = offset_bottom_rois[4] * spatial_scale;

        // Force malformed ROIs to be 1x1
        auto roi_width  = std::max(roi_end_w - roi_start_w, (float)1.);
        auto roi_height = std::max(roi_end_h - roi_start_h, (float)1.);
        auto bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
        auto bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : static_cast<int>(std::ceil(roi_height / pooled_height));  // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(std::ceil(roi_width / pooled_width));

        // We do average (integral) pooling inside a bin
        const int count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

        // we want to precalculate indices and weights shared by all channels,
        // this is the key point of optimization
        std::vector<PreCalc> pre_calc(roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
        pre_calc_for_bilinear_interpolate(height, width, pooled_height, pooled_width, roi_bin_grid_h, roi_bin_grid_w,
                                          roi_start_h, roi_start_w, bin_size_h, bin_size_w, roi_bin_grid_h,
                                          roi_bin_grid_w, pre_calc);

        for (int c = 0; c < channels; c++) {
            int index_n_c = index_n + c * pooled_width * pooled_height;
            const float *offset_bottom_data =
                input_data + static_cast<int>((roi_batch_ind * channels + c) * height * width);
            int pre_calc_index = 0;

            for (int ph = 0; ph < pooled_height; ph++) {
                for (int pw = 0; pw < pooled_width; pw++) {
                    int index = index_n_c + ph * pooled_width + pw;

                    float output_val = 0.;
                    if (layer_param->mode == "avg") {  // avg pooling
                        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                                PreCalc pc = pre_calc[pre_calc_index];
                                output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                                              pc.w2 * offset_bottom_data[pc.pos2] +
                                              pc.w3 * offset_bottom_data[pc.pos3] + pc.w4 * offset_bottom_data[pc.pos4];

                                pre_calc_index += 1;
                            }
                        }
                        output_val /= count;
                    } else {  // max pooling
                        bool max_flag = false;
                        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                                PreCalc pc = pre_calc[pre_calc_index];
                                float val  = std::max(std::max(std::max(pc.w1 * offset_bottom_data[pc.pos1],
                                                                        pc.w2 * offset_bottom_data[pc.pos2]),
                                                               pc.w3 * offset_bottom_data[pc.pos3]),
                                                      pc.w4 * offset_bottom_data[pc.pos4]);
                                if (!max_flag) {
                                    output_val = val;
                                    max_flag   = true;
                                } else {
                                    output_val = std::max(output_val, val);
                                }

                                pre_calc_index += 1;
                            }
                        }
                    }

                    output_data[index] = output_val;
                }  // for pw
            }      // for ph
        }          // for c
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(RoiAlign, LAYER_ROI_ALIGN);

}  // namespace TNN_NS
