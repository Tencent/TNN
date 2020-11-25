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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"
namespace TNN_NS {

static inline bool CheckInputOutputSizeSame(int input_height, int input_width,
            int output_height, int output_width) {
    return input_height == output_height && input_width == output_width;
}

// nearest interpolate function
static inline int upsample_nearest2d(float *output_data, const float *input_data, int input_height, int input_width,
                                     int output_height, int output_width, int channels, bool align_corners) {
    // special case: just copy
    if (CheckInputOutputSizeSame(input_height, input_width, output_height, output_width)) {
        if (output_data != input_data) {
            memcpy(output_data, input_data, channels * input_height * input_width * sizeof(float));
        }
        return 0;
    }

    const float height_scale = (float)input_height / (float)output_height;
    const float width_scale  = (float)input_width / (float)output_width;

    OMP_PARALLEL_FOR_
    for (int i = 0; i < channels; ++i) {
        int output_index  = i * output_height * output_width;
        int input_index_i = i * input_height * input_width;
        for (int j = 0; j < output_height; ++j) {
            int scaled_j      = static_cast<int>(j * height_scale);
            int input_index_j = input_index_i + scaled_j * input_width;
            for (int u = 0; u < output_width; ++u) {
                int scaled_u                = static_cast<int>(u * width_scale);
                output_data[output_index++] = input_data[input_index_j + scaled_u];
            }
        }
    }

    return 0;
}

// bilinear interpolate function
static inline int upsample_bilinear2d(float *output_data, const float *input_data, int input_height, int input_width,
                                      int output_height, int output_width, int channels, bool align_corners) {
    // special case: just copy
    if (CheckInputOutputSizeSame(input_height, input_width, output_height, output_width)) {
        if (output_data != input_data) {
            memcpy(output_data, input_data, channels * input_height * input_width * sizeof(float));
        }
        return 0;
    }

    // align corners option from pytorch
    if (align_corners) {
        const float rheight = (output_height > 1) ? (float)(input_height - 1) / (output_height - 1) : 0.f;
        const float rwidth  = (output_width > 1) ? (float)(input_width - 1) / (output_width - 1) : 0.f;
        OMP_PARALLEL_FOR_
        for (int h2 = 0; h2 < output_height; ++h2) {
            const float h1r = rheight * h2;

            const int h1         = static_cast<int>(h1r);
            const int h1p        = (h1 < input_height - 1) ? 1 : 0;
            const float h1lambda = h1r - h1;
            const float h0lambda = (float)1. - h1lambda;
            for (int w2 = 0; w2 < output_width; ++w2) {
                const float w1r      = rwidth * w2;
                const int w1         = static_cast<int>(w1r);
                const int w1p        = (w1 < input_width - 1) ? 1 : 0;
                const float w1lambda = w1r - w1;
                const float w0lambda = (float)1. - w1lambda;
                const float *Xdata   = &(input_data[h1 * input_width + w1]);
                float *Ydata         = &(output_data[h2 * output_width + w2]);
                for (int c = 0; c < channels; ++c) {
                    Ydata[0] =
                        h0lambda * (w0lambda * Xdata[0] + w1lambda * Xdata[w1p]) +
                        h1lambda * (w0lambda * Xdata[h1p * input_width] + w1lambda * Xdata[h1p * input_width + w1p]);
                    Xdata += input_width * input_height;
                    Ydata += output_width * output_height;
                }
            }
        }
    } else {
        const float rheight = (output_height > 1) ? (float)(input_height) / (output_height) : 0.f;
        const float rwidth  = (output_width > 1) ? (float)(input_width) / (output_width) : 0.f;

        OMP_PARALLEL_FOR_
        for (int h2 = 0; h2 < output_height; ++h2) {
            float h1r     = static_cast<float>(rheight * (h2 + 0.5) - 0.5);
            h1r           = h1r >= 0 ? h1r : 0;
            const int h1  = static_cast<int>(h1r);
            const int h1p = (h1 < input_height - 1) ? 1 : 0;

            const float h1lambda = h1r - h1;
            const float h0lambda = (float)1. - h1lambda;

            for (int w2 = 0; w2 < output_width; ++w2) {
                float w1r = static_cast<float>(rwidth * (w2 + 0.5) - 0.5);
                w1r       = w1r >= 0 ? w1r : 0;

                const int w1            = static_cast<int>(w1r);
                const int w1p           = (w1 < input_width - 1) ? 1 : 0;
                const float w1lambda    = w1r - w1;
                const float w0lambda    = (float)1. - w1lambda;
                const float *x_data_ptr = &(input_data[h1 * input_width + w1]);
                float *y_data_ptr       = &(output_data[h2 * output_width + w2]);
                for (int c = 0; c < channels; ++c) {
                    y_data_ptr[0] = h0lambda * (w0lambda * x_data_ptr[0] + w1lambda * x_data_ptr[w1p]) +
                                    h1lambda * (w0lambda * x_data_ptr[h1p * input_width] +
                                                w1lambda * x_data_ptr[h1p * input_width + w1p]);
                    x_data_ptr += input_width * input_height;
                    y_data_ptr += output_width * output_height;
                }
            }
        }
    }

    return 0;
}

// cubic interpolate weights
template <typename T>
static void GetCubicWeights(float coor, T coeffs[4]) {
    // opencv uses -0.75
    static const float A = -0.75f;
    float x = coor - std::floor(coor);

    coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
    coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
    coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

// cubic interpolate function
template <bool align_corners>
static void upsample_cubic2d_impl(float *dst, const float *src, int sh, int sw,
                                      int dh, int dw, int channels) {
    const float h_scale = (dh > 1) ? (align_corners ? (float)(sh - 1) / (dh - 1)
                                                : (float)(sh) / (dh)) : 0.f;
    const float w_scale = (dw > 1) ? (align_corners ? (float)(sw - 1) / (dw - 1)
                                                : (float)(sw) / (dw)) : 0.f;
#define Clip(x,X) ( (x) >=0 ? ((x)<(X)?(x):((X)-1)) : 0 )
#define SrcValueAt(c, h, w) (src[c*sh*sw+(Clip(h,sh))*sw+(Clip(w,sw))])

        _Pragma("omp parallel for")
        for (int h2 = 0; h2 < dh; ++h2) {
            float h1 = static_cast<float>(align_corners ? h_scale * h2 : h_scale * (h2 + 0.5) - 0.5);
            int hh = std::floor(h1);
            float wy[4];
            GetCubicWeights(h1, wy);
            for (int w2 = 0; w2 < dw; ++w2) {
                float w1 = static_cast<float>(align_corners? w_scale * w2 : w_scale * (w2 + 0.5) - 0.5);
                int ww = std::floor(w1);
                float wx[4];
                GetCubicWeights(w1, wx);
                for (int c = 0; c < channels; ++c) {
                    float src_arr[4][4] = {
                        {SrcValueAt(c, hh-1, ww-1), SrcValueAt(c, hh-1, ww), SrcValueAt(c, hh-1, ww+1), SrcValueAt(c, hh-1, ww+2)},
                        {SrcValueAt(c, hh+0, ww-1), SrcValueAt(c, hh+0, ww), SrcValueAt(c, hh+0, ww+1), SrcValueAt(c, hh+0, ww+2)},
                        {SrcValueAt(c, hh+1, ww-1), SrcValueAt(c, hh+1, ww), SrcValueAt(c, hh+1, ww+1), SrcValueAt(c, hh+1, ww+2)},
                        {SrcValueAt(c, hh+2, ww-1), SrcValueAt(c, hh+2, ww), SrcValueAt(c, hh+2, ww+1), SrcValueAt(c, hh+2, ww+2)}
                    };
                    float vals[4];
                    vals[0] = wx[0]*src_arr[0][0] + wx[1]*src_arr[0][1] + wx[2]*src_arr[0][2] + wx[3]*src_arr[0][3];
                    vals[1] = wx[0]*src_arr[1][0] + wx[1]*src_arr[1][1] + wx[2]*src_arr[1][2] + wx[3]*src_arr[1][3];
                    vals[2] = wx[0]*src_arr[2][0] + wx[1]*src_arr[2][1] + wx[2]*src_arr[2][2] + wx[3]*src_arr[2][3];
                    vals[3] = wx[0]*src_arr[3][0] + wx[1]*src_arr[3][1] + wx[2]*src_arr[3][2] + wx[3]*src_arr[3][3];

                    float sum = wy[0]*vals[0] + wy[1]*vals[1] + wy[2]*vals[2] + wy[3]*vals[3];
                    dst[(c * dh + h2) * dw + w2] = sum;
                }
            }
        }
#undef Clip
#undef SrcValueAt
}

static inline int upsample_cubic2d(float *output_data, const float *input_data, int input_height, int input_width,
                                      int output_height, int output_width, int channels, bool align_corners) {
    if (align_corners)
        upsample_cubic2d_impl<true>(output_data, input_data, input_height,
                     input_width, output_height, output_width, channels);
    else
        upsample_cubic2d_impl<false>(output_data, input_data, input_height,
                     input_width, output_height, output_width, channels);

    return 0;
}

DECLARE_CPU_ACC(Upsample, LAYER_UPSAMPLE);

Status CpuUpsampleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuUpsampleLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<UpsampleLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: UpsampleLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto dims_input   = input_blob->GetBlobDesc().dims;
    auto dims_output  = output_blob->GetBlobDesc().dims;

    auto input_width = dims_input[3], input_height = dims_input[2];
    auto output_width = dims_output[3], output_height = dims_output[2], output_channel = dims_output[1];

    float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
    float *output_data = static_cast<float *>(output_blob->GetHandle().base);

    if (param->mode == 1) {  // nearest
        upsample_nearest2d(output_data, input_data, input_height, input_width, output_height, output_width,
                           output_channel, (bool)param->align_corners);
    } else if (param->mode == 2) {  // bilinear/linear
        upsample_bilinear2d(output_data, input_data, input_height, input_width, output_height, output_width,
                            output_channel, (bool)param->align_corners);

    } else if (param->mode == 3) { // cubic
        upsample_cubic2d(output_data, input_data, input_height, input_width, output_height, output_width,
                            output_channel, (bool)param->align_corners);
    }else {
        LOGE("Error: Upsample dont support resize type\n");
        return Status(TNNERR_MODEL_ERR, "Error: Upsample dont support resize type");
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(Upsample, LAYER_UPSAMPLE);

}  // namespace TNN_NS
