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

#include "tnn/device/x86/acc/x86_upsample_layer_acc.h"

#include "tnn/core/blob_int8.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/omp_utils.h"
#include "tnn/device/x86/acc/compute/x86_compute_int8.h"

namespace TNN_NS {

static inline bool CheckInputOutputSizeSame(int input_height, int input_width, int output_height, int output_width) {
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

static inline void get_bilinear_coeffs(float *h_coeffs_ptr, float *w_coeffs_ptr, int ih, int iw, int oh, int ow,
                                       bool align_corners) {
    if (align_corners) {
        const float rheight = (oh > 1) ? (float)(ih - 1) / (oh - 1) : 0.f;
        const float rwidth  = (ow > 1) ? (float)(iw - 1) / (ow - 1) : 0.f;
        for (int h = 0; h < oh; ++h) {
            h_coeffs_ptr[h] = h * rheight;
        }
        for (int w = 0; w < ow; ++w) {
            w_coeffs_ptr[w] = w * rwidth;
        }
    } else {
        const float rheight = (oh > 1) ? (float)(ih) / (oh) : 0.f;
        const float rwidth  = (ow > 1) ? (float)(iw) / (ow) : 0.f;
        for (int h = 0; h < oh; ++h) {
            h_coeffs_ptr[h] = rheight * (h + 0.5) - 0.5;
            h_coeffs_ptr[h] = h_coeffs_ptr[h] >= 0 ? h_coeffs_ptr[h] : 0;
        }
        for (int w = 0; w < ow; ++w) {
            w_coeffs_ptr[w] = rwidth * (w + 0.5) - 0.5;
            w_coeffs_ptr[w] = w_coeffs_ptr[w] >= 0 ? w_coeffs_ptr[w] : 0;
        }
    }
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

    RawBuffer h_coeffs(output_height * sizeof(float));
    RawBuffer w_coeffs(output_width * sizeof(float));
    auto h_coeffs_ptr = h_coeffs.force_to<float *>();
    auto w_coeffs_ptr = w_coeffs.force_to<float *>();

    get_bilinear_coeffs(h_coeffs_ptr, w_coeffs_ptr, input_height, input_width, output_height, output_width, align_corners);

    OMP_PARALLEL_FOR_
    for (int h2 = 0; h2 < output_height; ++h2) {
        const float h1r      = h_coeffs_ptr[h2];
        const int h1         = h1r;
        const int h1p        = (h1 < input_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = (float)1. - h1lambda;
        for (int w2 = 0; w2 < output_width; ++w2) {
            const float w1r      = w_coeffs_ptr[w2];
            const int w1         = w1r;
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

        OMP_PARALLEL_FOR_
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

static inline bool need_do_scale(const float *scale, int len) {
    for (int i = 0; i < len; ++i) {
        if (fabs(scale[i] - 1.0) > 0.0078125) {
            return true;
        }
    }
    return false;
}

X86UpsampleLayerAcc::~X86UpsampleLayerAcc() {}

Status X86UpsampleLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<UpsampleLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: UpsampleLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto dims_input   = input_blob->GetBlobDesc().dims;
    auto dims_output  = output_blob->GetBlobDesc().dims;

    auto batch       = dims_input[0];
    auto channel     = dims_input[1];
    auto input_width = dims_input[3], input_height = dims_input[2];
    auto output_width = dims_output[3], output_height = dims_output[2];
    auto input_plane  = input_width * input_height * channel;
    auto output_plane = output_width * output_height * channel;

    DataType data_type = output_blob->GetBlobDesc().data_type;

    float *input_data  = handle_ptr<float *>(input_blob->GetHandle());
    float *output_data = handle_ptr<float *>(output_blob->GetHandle());

    RawBuffer buffer_scale_;
    bool do_scale_;
    if (data_type == DATA_TYPE_INT8) {
        int total_byte_size = ROUND_UP(dims_output[1], 4) * sizeof(float);
        input_plane  = ROUND_UP(dims_input[1], 4) * DimsVectorUtils::Count(dims_input, 2);
        output_plane = ROUND_UP(dims_output[1], 4) * DimsVectorUtils::Count(dims_output, 2);;

        auto input_resource  = reinterpret_cast<BlobInt8 *>(inputs[0])->GetIntResource();
        auto output_resource = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource();
        const float *i_scale = input_resource->scale_handle.force_to<float *>();
        const float *o_scale = output_resource->scale_handle.force_to<float *>();
        int scale_len_i      = input_resource->scale_handle.GetDataCount();
        int scale_len_o      = output_resource->scale_handle.GetDataCount();

        if (buffer_scale_.GetBytesSize() < total_byte_size) {
            buffer_scale_ = RawBuffer(total_byte_size);
        }
        float *temp_ptr = buffer_scale_.force_to<float *>();
        for (int i = 0; i < dims_output[1]; i++) {
            int scale_idx_i = scale_len_i == 1 ? 0 : i;
            int scale_idx_o = scale_len_o == 1 ? 0 : i;
            if (o_scale[scale_idx_o] >= FLT_MIN)
                temp_ptr[i] = i_scale[scale_idx_i] / o_scale[scale_idx_o];
            else
                temp_ptr[i] = 0.0;
        }
        do_scale_ = need_do_scale(temp_ptr, dims_output[1]);

        auto oc_4 = UP_DIV(dims_output[1], 4);
        if (dims_input[2] == dims_output[2] && dims_input[3] == dims_output[3] && !do_scale_) {
            if (output_data != input_data) {
                memcpy(output_data, input_data, batch * input_plane * DataTypeUtils::GetBytesSize(data_type));
            }
        } else if (param->mode == 1) {
            for (int b = 0; b < batch; ++b) {
                auto output_b = reinterpret_cast<int8_t *>(output_data) + b * output_plane;
                auto input_b  = reinterpret_cast<int8_t *>(input_data) + b * input_plane;
                if (do_scale_)
                    X86UpsampleNearest2D<true>(output_b, input_b, dims_input[2], dims_input[3], dims_output[2],
                                               dims_output[3], oc_4, buffer_scale_.force_to<float *>());
                else
                    X86UpsampleNearest2D<false>(output_b, input_b, dims_input[2], dims_input[3], dims_output[2],
                                                dims_output[3], oc_4, buffer_scale_.force_to<float *>());
            }
        } else if (param->mode == 2) {
            if (do_scale_)
                X86UpsampleBilinear2D<true>(reinterpret_cast<int8_t *>(output_data),
                                            reinterpret_cast<int8_t *>(input_data), batch, dims_input[2], dims_input[3],
                                            dims_output[2], dims_output[3], oc_4, (bool)param->align_corners,
                                            buffer_scale_.force_to<float *>());
            else
                X86UpsampleBilinear2D<false>(reinterpret_cast<int8_t *>(output_data),
                                             reinterpret_cast<int8_t *>(input_data), batch, dims_input[2], dims_input[3],
                                             dims_output[2], dims_output[3], oc_4, (bool)param->align_corners,
                                             buffer_scale_.force_to<float *>());
        } else {
            LOGE("Error: Not supported mode for x86 int8 upsample\n");
            return Status(TNNERR_PARAM_ERR, "Error: Not supported mode for x86 int8 upsample");
        }
    } else if (data_type == DATA_TYPE_FLOAT) {
        if (param->mode == 1) {  // nearest
            for (int b = 0; b < batch; ++b) {
                upsample_nearest2d(output_data + b * output_plane, input_data + b * input_plane, input_height, input_width,
                                output_height, output_width, channel, (bool)param->align_corners);
            }
        } else if (param->mode == 2) {  // bilinear/linear
            for (int b = 0; b < batch; ++b) {
                upsample_bilinear2d(output_data + b * output_plane, input_data + b * input_plane, input_height, input_width,
                                    output_height, output_width, channel, (bool)param->align_corners);
            }
        } else if (param->mode == 3) { // cubic
            for (int b = 0; b < batch; ++b) {
                upsample_cubic2d(output_data + b * output_plane, input_data + b * input_plane, input_height, input_width,
                                output_height, output_width, channel, (bool)param->align_corners);
            }
        } else {
            LOGE("Error: Not supported mode for x86 float upsample\n");
            return Status(TNNERR_MODEL_ERR, "Error: Not supported mode for x86 float upsample");
        }
    } else {
        LOGE("Error: Not supported data type for upsample\n");
        return Status(TNNERR_LAYER_ERR, "Error: Not supported data type for upsample");
    }

    return TNN_OK;
}

REGISTER_X86_ACC(Upsample, LAYER_UPSAMPLE);

}  // namespace TNN_NS
