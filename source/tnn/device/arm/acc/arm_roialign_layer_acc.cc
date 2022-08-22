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
// under the License is distributed on an "AS IS" BASIS, WITHOUfloat WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "math.h"
#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(RoiAlign, LAYER_ROIALIGN);

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

struct CalRoiParam {
    int n_rois;
    int channels;
    int pooled_height;
    int pooled_width;
    int orig_height;
    int orig_width;
    int num_roi_cols;
    int mode;
    int sampling_ratio;
    float spatial_scale;

    float roi_start_h;
    float roi_start_w;
    float bin_size_h;
    float bin_size_w;
    int roi_bin_grid_h;
    int roi_bin_grid_w;
};

static void PreCalcForBilinearInterpolate(const CalRoiParam &param, std::vector<PreCalc> &pre_calc, int scale = 1) {
    const int orig_height   = param.orig_height;
    const int orig_width    = param.orig_width;
    const int pooled_height = param.pooled_height;
    const int pooled_width  = param.pooled_width;
    const int iy_upper      = param.roi_bin_grid_h;
    const int ix_upper      = param.roi_bin_grid_w;

    int pre_calc_index = 0;
    for (int ph = 0; ph < pooled_height; ph++) {
        float y_h = param.roi_start_h + ph * param.bin_size_h;
        for (int pw = 0; pw < pooled_width; pw++) {
            float x_w = param.roi_start_w + pw * param.bin_size_w;
            for (int iy = 0; iy < iy_upper; iy++) {
                const float yy = y_h + (iy + .5f) * param.bin_size_h / static_cast<float>(iy_upper);
                for (int ix = 0; ix < ix_upper; ix++) {
                    const float xx = x_w + (ix + .5f) * param.bin_size_w / static_cast<float>(ix_upper);

                    float x = xx;
                    float y = yy;

                    if (y < -1.0 || y > orig_height || x < -1.0 || x > orig_width) {
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

                    if (y_low >= orig_height - 1) {
                        y_high = y_low = orig_height - 1;
                        y              = (float)y_low;
                    } else {
                        y_high = y_low + 1;
                    }

                    if (x_low >= orig_width - 1) {
                        x_high = x_low = orig_width - 1;
                        x              = (float)x_low;
                    } else {
                        x_high = x_low + 1;
                    }

                    float ly = y - y_low;
                    float lx = x - x_low;
                    float hy = static_cast<float>(1.) - ly;
                    float hx = static_cast<float>(1.) - lx;
                    float w1 = hy * hx;
                    float w2 = hy * lx;
                    float w3 = ly * hx;
                    float w4 = ly * lx;

                    // save weights and indeces
                    PreCalc pc;
                    pc.pos1 = y_low * orig_width + x_low;
                    pc.pos2 = y_low * orig_width + x_high;
                    pc.pos3 = y_high * orig_width + x_low;
                    pc.pos4 = y_high * orig_width + x_high;
                    if (scale != 1) {
                        pc.pos1 *= scale;
                        pc.pos2 *= scale;
                        pc.pos3 *= scale;
                        pc.pos4 *= scale;
                    }
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

static Status CalculateRoiAlignNCHW(CalRoiParam &param, const float *bottom_data, const float *bottom_rois,
                                    const int *batch_indices_ptr, float *top_data) {
    int channels      = param.channels;
    int pooled_height = param.pooled_height;
    int pooled_width  = param.pooled_width;
    int orig_height   = param.orig_height;
    int orig_width    = param.orig_width;

    for (int n = 0; n < param.n_rois; ++n) {
        int index_n = n * channels * pooled_width * pooled_height;

        const float *bottom_rois_n = bottom_rois + n * param.num_roi_cols;
        const auto roi_batch_ind   = batch_indices_ptr[n];
        const float *bottom_data_n = bottom_data + roi_batch_ind * channels * orig_height * orig_width;

        param.roi_start_w = bottom_rois_n[0] * param.spatial_scale;
        param.roi_start_h = bottom_rois_n[1] * param.spatial_scale;

        float roi_end_w  = bottom_rois_n[2] * param.spatial_scale;
        float roi_end_h  = bottom_rois_n[3] * param.spatial_scale;
        float roi_width  = std::max(roi_end_w - param.roi_start_w, (float)1.);
        float roi_height = std::max(roi_end_h - param.roi_start_h, (float)1.);
        param.bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
        param.bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

        if (param.sampling_ratio > 0) {
            param.roi_bin_grid_h = param.sampling_ratio;
            param.roi_bin_grid_w = param.sampling_ratio;
        } else {
            param.roi_bin_grid_h = static_cast<int>(std::ceil(roi_height / pooled_height));
            param.roi_bin_grid_w = static_cast<int>(std::ceil(roi_width / pooled_width));
        }

        const int roi_dot_count = param.roi_bin_grid_h * param.roi_bin_grid_w;

        std::vector<PreCalc> pre_calc(roi_dot_count * pooled_width * pooled_height);
        PreCalcForBilinearInterpolate(param, pre_calc);

        for (int c = 0; c < channels; c++) {
            int index_c                = index_n + c * pooled_width * pooled_height;
            const float *bottom_data_c = bottom_data_n + c * orig_height * orig_width;
            int pre_calc_index         = 0;

            for (int ph = 0; ph < pooled_height; ph++) {
                for (int pw = 0; pw < pooled_width; pw++) {
                    int index = index_c + ph * pooled_width + pw;

                    float output_val = 0.;
                    if (param.mode == 1) {  // avg pooling
                        for (int iy = 0; iy < param.roi_bin_grid_h; iy++) {
                            for (int ix = 0; ix < param.roi_bin_grid_w; ix++) {
                                PreCalc pc = pre_calc[pre_calc_index];
                                output_val += pc.w1 * bottom_data_c[pc.pos1] + pc.w2 * bottom_data_c[pc.pos2] +
                                              pc.w3 * bottom_data_c[pc.pos3] + pc.w4 * bottom_data_c[pc.pos4];

                                pre_calc_index += 1;
                            }
                        }
                        output_val /= roi_dot_count;
                    } else {  // max pooling
                        bool max_flag = false;
                        for (int iy = 0; iy < param.roi_bin_grid_h; iy++) {
                            for (int ix = 0; ix < param.roi_bin_grid_w; ix++) {
                                PreCalc pc = pre_calc[pre_calc_index];
                                float val  = std::max(
                                     std::max(std::max(pc.w1 * bottom_data_c[pc.pos1], pc.w2 * bottom_data_c[pc.pos2]),
                                              pc.w3 * bottom_data_c[pc.pos3]),
                                     pc.w4 * bottom_data_c[pc.pos4]);
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

                    top_data[index] = output_val;
                }
            }
        }
    }

    return TNN_OK;
}

static Status CalculateRoiAlign(CalRoiParam &param, const float *bottom_data, const float *bottom_rois,
                                const int *batch_indices_ptr, float *top_data) {
    int channels      = ROUND_UP(param.channels, 4);
    int pooled_height = param.pooled_height;
    int pooled_width  = param.pooled_width;
    int orig_height   = param.orig_height;
    int orig_width    = param.orig_width;

    for (int n = 0; n < param.n_rois; ++n) {
        int index_n = n * channels * pooled_width * pooled_height;

        const float *bottom_rois_n = bottom_rois + n * ROUND_UP(param.num_roi_cols, 4);
        const auto roi_batch_ind   = batch_indices_ptr[n * 4];
        const float *bottom_data_n = bottom_data + roi_batch_ind * channels * orig_height * orig_width;

        param.roi_start_w = bottom_rois_n[0] * param.spatial_scale;
        param.roi_start_h = bottom_rois_n[1] * param.spatial_scale;

        float roi_end_w  = bottom_rois_n[2] * param.spatial_scale;
        float roi_end_h  = bottom_rois_n[3] * param.spatial_scale;
        float roi_width  = std::max(roi_end_w - param.roi_start_w, (float)1.);
        float roi_height = std::max(roi_end_h - param.roi_start_h, (float)1.);
        param.bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
        param.bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

        if (param.sampling_ratio > 0) {
            param.roi_bin_grid_h = param.sampling_ratio;
            param.roi_bin_grid_w = param.sampling_ratio;
        } else {
            param.roi_bin_grid_h = static_cast<int>(std::ceil(roi_height / pooled_height));
            param.roi_bin_grid_w = static_cast<int>(std::ceil(roi_width / pooled_width));
        }

        const int roi_dot_count = param.roi_bin_grid_h * param.roi_bin_grid_w;

        std::vector<PreCalc> pre_calc(roi_dot_count * pooled_width * pooled_height);
        PreCalcForBilinearInterpolate(param, pre_calc, 4);

        for (int c = 0; c < channels; c += 4) {
            int index_c                = index_n + c * pooled_width * pooled_height;
            const float *bottom_data_c = bottom_data_n + c * orig_height * orig_width;
            int pre_calc_index         = 0;

            for (int ph = 0; ph < pooled_height; ph++) {
                for (int pw = 0; pw < pooled_width; pw++) {
                    int index = index_c + (ph * pooled_width + pw) * 4;

                    Float4 output_val;
                    if (param.mode == 1) {
                        // avg pooling
                        output_val = Float4(0.);
                        for (int iy = 0; iy < param.roi_bin_grid_h; iy++) {
                            for (int ix = 0; ix < param.roi_bin_grid_w; ix++) {
                                PreCalc pc = pre_calc[pre_calc_index];
                                output_val = output_val + Float4::load(bottom_data_c + pc.pos1) * pc.w1;
                                output_val = output_val + Float4::load(bottom_data_c + pc.pos2) * pc.w2;
                                output_val = output_val + Float4::load(bottom_data_c + pc.pos3) * pc.w3;
                                output_val = output_val + Float4::load(bottom_data_c + pc.pos4) * pc.w4;
                                pre_calc_index += 1;
                            }
                        }
                        output_val = output_val * (1.0 / roi_dot_count);
                    } else {
                        // max pooling
                        output_val = Float4(-FLT_MAX);
                        for (int iy = 0; iy < param.roi_bin_grid_h; iy++) {
                            for (int ix = 0; ix < param.roi_bin_grid_w; ix++) {
                                PreCalc pc = pre_calc[pre_calc_index];
                                output_val = Float4::max(output_val, Float4::load(bottom_data_c + pc.pos1) * pc.w1);
                                output_val = Float4::max(output_val, Float4::load(bottom_data_c + pc.pos2) * pc.w2);
                                output_val = Float4::max(output_val, Float4::load(bottom_data_c + pc.pos3) * pc.w3);
                                output_val = Float4::max(output_val, Float4::load(bottom_data_c + pc.pos4) * pc.w4);
                                pre_calc_index += 1;
                            }
                        }
                    }

                    Float4::save(top_data + index, output_val);
                }
            }
        }
    }

    return TNN_OK;
}

Status ArmRoiAlignLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<RoiAlignLayerParam *>(param_);
    if (!param) {
        LOGE("Error: RoiAlignLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: RoiAlignLayerParam is nil");
    }
    if (inputs.size() < 3) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "Concat layer's inputs size must >= 3");
    }

    auto input_blob    = inputs[0];
    auto rois          = inputs[1];
    auto batch_indices = inputs[2];
    auto output_blob   = outputs[0];

    auto input_dims  = input_blob->GetBlobDesc().dims;
    auto rois_dims   = rois->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;

    auto *input_ptr         = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));
    auto *rois_ptr          = reinterpret_cast<float *>(GetBlobHandlePtr(rois->GetHandle()));
    auto *batch_indices_ptr = reinterpret_cast<int *>(GetBlobHandlePtr(batch_indices->GetHandle()));
    auto *output_ptr        = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));

    CalRoiParam cal_param = {
        .n_rois         = output_dims[0],
        .channels       = output_dims[1],
        .pooled_height  = output_dims[2],
        .pooled_width   = output_dims[3],
        .orig_height    = input_dims[2],
        .orig_width     = input_dims[3],
        .num_roi_cols   = rois_dims[1],
        .mode           = param->mode,
        .sampling_ratio = param->sampling_ratio,
        .spatial_scale  = param->spatial_scale,
    };

    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        if (inputs[0]->GetBlobDesc().data_format == DATA_FORMAT_NCHW) {
            return CalculateRoiAlignNCHW(cal_param, input_ptr, rois_ptr, batch_indices_ptr, output_ptr);
        } else {
            return CalculateRoiAlign(cal_param, input_ptr, rois_ptr, batch_indices_ptr, output_ptr);
        }
    } else {
        return Status(TNNERR_LAYER_ERR, "Unsupported data type in roi align");
    }

    return TNNERR_LAYER_ERR;
}

REGISTER_ARM_ACC(RoiAlign, LAYER_ROIALIGN)
REGISTER_ARM_LAYOUT(LAYER_ROIALIGN, DATA_FORMAT_NC4HW4)
// REGISTER_ARM_LAYOUT(LAYER_ROIALIGN, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
