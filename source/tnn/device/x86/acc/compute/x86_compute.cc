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

#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/device/x86/acc/Float8.h"
#include "tnn/device/x86/acc/Float4.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/omp_utils.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <type_traits>
#include <immintrin.h>

#include "jit/cblas.h"

namespace TNN_NS {

Status X86_IM2COL(float* src, int channel, int height, int width, int kernelh, int kernelw, int padl, int padr,
                  int padt, int padb, int strideh, int stridew, int dilationh, int dilationw, float* dst) {
    int height_col   = (height + padt + padb - dilationh * (kernelh - 1) - 1) / strideh + 1;
    int width_col    = (width + padl + padr - dilationw * (kernelw - 1) - 1) / stridew + 1;
    int channels_col = channel * kernelh * kernelw;

    // im2col
    for (int c = 0; c < channels_col; c++) {
        int w_offset = c % kernelw;
        int h_offset = (c / kernelw) % kernelh;
        int c_im     = c / kernelh / kernelw;

        int h_base = h_offset * dilationh - padt;
        int w_base = w_offset * dilationw - padl;

        int h_base_start = MAX(0, (UP_DIV(-h_base, strideh)));
        int h_base_end   = MIN(height_col, UP_DIV(height - h_base, strideh));
        int w_base_start = MAX(0, (UP_DIV(-w_base, stridew)));
        int w_base_end   = MIN(width_col, UP_DIV(width - w_base, stridew));

        auto src_c = src + c_im * height * width;
        auto dst_c = dst + c * height_col * width_col;

        memset(dst_c, 0, h_base_start * width_col * sizeof(float));
        for (int h = h_base_start; h < h_base_end; h++) {
            int h_pad = h_base + h * strideh;

            auto src_h = src_c + h_pad * width;
            auto dst_h = dst_c + h * width_col;

            for (int w = 0; w < w_base_start; w++) {
                dst_h[w] = 0;
            }
            for (int w = w_base_start; w < w_base_end; w++) {
                int w_pad = w_base + w * stridew;
                dst_h[w]  = src_h[w_pad];
            }
            for (int w = w_base_end; w < width_col; w++) {
                dst_h[w] = 0;
            }
        }
        memset(dst_c + h_base_end * width_col, 0, (height_col - h_base_end) * width_col * sizeof(float));
    }

    return TNN_OK;
}

Status X86_COL2IM(float* src, int channels, int height, int width, int kernelh, int kernelw, int padh, int padw,
                  int strideh, int stridew, int dilationh, int dilationw, int output_height, int output_width, float* dst) {
    for (int c = 0; c < channels; ++c) {
        auto dst_c = dst + c * output_height * output_width;
        auto src_c = src + c * kernelh * kernelw * width * height;
        memset(dst_c, 0, output_height * output_width * 4);
        for (int dh = 0; dh < height; ++dh) {
            for (int dw = 0; dw < width; ++dw) {
                int src_start_y = dh * strideh - padh;
                int src_start_x = dw * stridew - padw;
                int sfy = MAX(0, UP_DIV(-src_start_y, dilationh));
                int efy = MIN(kernelh, UP_DIV(output_height - src_start_y, dilationh));
                int sfx = MAX(0, UP_DIV(-src_start_x, dilationw));
                int efx = MIN(kernelw, UP_DIV(output_width - src_start_x, dilationw));

                auto dst_start = dst_c + src_start_y * output_width + src_start_x;
                auto src_start = src_c + dh * width + dw;

                for (int fy = sfy; fy < efy; ++fy) {
                    auto dst_y = dst_start + fy * dilationh * output_width;
                    auto src_y = src_start + fy * kernelw * height * width;
                    for (int fx = sfx; fx < efx; ++fx) {
                        dst_y[fx * dilationw] += src_y[fx * width * height];
                    }
                }
            }
        }
    }

    return TNN_OK;
}

Status X86_matrixMul(int m, int n, int k, const float *A, const float *B, float *C,
                     int has_bias, const float *bias, int activation_type) {

    if (ActivationType_None == activation_type) {
        // row major matrix A(m, k) x B(k, n) equals to :
        // col major matrix B(n, k) x A(k, m).
        float alpha = 1.0;
        float beta = 0.0;
        if (1 == has_bias){
            beta = 1.0;
            for (int mm = 0; mm < m; mm++) {
                for (int nn = 0; nn < n; nn++) {
                    C[mm * n + nn] = bias[mm];
                }
            }
        } 
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                    n, m, k, alpha, B, n, A, k, beta, C, n);
    } else {
        for (int mm = 0; mm < m; mm++) {
            for (int nn = 0; nn < n; nn++) {
                float tmp = 0.f;
                for (int kk = 0; kk < k; kk++) {
                    tmp += A[mm * k + kk] * B[kk * n + nn];
                }
                if (has_bias) tmp += bias[mm];
                if (activation_type == ActivationType_ReLU) tmp = std::max(0.f, tmp);
                C[mm * n + nn] = tmp;
            }
        }
    }
    return TNN_OK;
}

// max pooling can use heap
Status X86_MAX_POOLING(float *input, float *output, DimsVector input_dim, DimsVector output_dim,
                       int stride_h, int stride_w, int kernel_h, int kernel_w, int pad_h, int pad_w) {
    
    auto input_width = input_dim[3], input_height = input_dim[2], input_channel = input_dim[1];
    auto output_width = output_dim[3], output_height = output_dim[2], output_channel = output_dim[1];

    for (int b = 0; b < input_dim[0]; b++) {
        for (int c = 0; c < output_channel; c++) {
            float* input_data  = input + (b * input_channel + c) * input_height * input_width;
            float* output_data = output + (b * output_channel + c) * output_height * output_width;
            for (int h = 0; h < output_height; h++) {
                for (int w = 0; w < output_width; w++) {
                    // if use heap, build for first try
                    int h_start = h * stride_h - pad_h, h_end = h_start + kernel_h;
                    int w_start = w * stride_w - pad_w, w_end = w_start + kernel_w;
                    h_start = std::max(0, h_start); h_end = std::min(input_height, h_end);
                    w_start = std::max(0, w_start); w_end = std::min(input_width, w_end);
                    float tmp = input_data[h_start * input_width + w_start];
                    for (int hin = h_start; hin < h_end; hin++) {
                        for (int win = w_start; win < w_end; win++) {
                            tmp = std::max(tmp, input_data[hin * input_width + win]);
                        }
                    }
                    output_data[h * output_width + w] = tmp;
                }
            }
        }
    }
    return TNN_OK;
}

Status X86_AVERAGE_POOLING(float *input, float *output, DimsVector input_dim, DimsVector output_dim,
                           int stride_h, int stride_w, int kernel_h, int kernel_w, int pad_h, int pad_w) {
    
    auto input_width = input_dim[3], input_height = input_dim[2], input_channel = input_dim[1];
    auto output_width = output_dim[3], output_height = output_dim[2], output_channel = output_dim[1];

    for (int b = 0; b < input_dim[0]; b++) {
        for (int c = 0; c < output_channel; c++) {
            float* input_data  = input + (b * input_channel + c) * input_height * input_width;
            float* output_data = output + (b * output_channel + c) * output_height * output_width;
            for (int h = 0; h < output_height; h++) {
                for (int w = 0; w < output_width; w++) {
                    // if use heap, build for first try
                    int h_start = h * stride_h - pad_h, h_end = h_start + kernel_h;
                    int w_start = w * stride_w - pad_w, w_end = w_start + kernel_w;
                    h_start = std::max(0, h_start); h_end = std::min(input_height, h_end);
                    w_start = std::max(0, w_start); w_end = std::min(input_width, w_end);
                    auto kernel_count = (h_end - h_start) * (w_end - w_start);
                    float tmp = 0.f;
                    for (int hin = h_start; hin < h_end; hin++) {
                        for (int win = w_start; win < w_end; win++) {
                            tmp += input_data[hin * input_width + win];
                        }
                    }
                    output_data[h * output_width + w] = tmp / kernel_count;
                }
            }
        }
    }
    return TNN_OK;
}

/*
max pooling corner func, left/right/top/bottom
*/
template <class T, int pack_c>
void X86MaxPoolingCorner(const float* src, long iw, long ih, float* dst, long ow, long kw, long kh, long stride_w, long stride_h,
                      long pad_w, long pad_h, long l, long r, long t, long b) {
    for (long oy = t; oy < b; ++oy) {
        for (long ox = l; ox < r; ++ox) {
            T vmax(-FLT_MAX);

            const long srcOriginX = ox * stride_w - pad_w;
            const long srcOriginY = oy * stride_h - pad_h;
            const long kxs        = MAX(0, -srcOriginX);
            const long kxe        = MIN(kw, iw - srcOriginX);
            const long kys        = MAX(0, -srcOriginY);
            const long kye        = MIN(kh, ih - srcOriginY);
            const auto src_ptr    = src + (srcOriginY * iw + srcOriginX) * pack_c;
            auto dst_ptr          = dst + (oy * ow + ox) * pack_c;

            for (long ky = kys; ky < kye; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * pack_c;
                for (long kx = kxs; kx < kxe; kx++) {
                    vmax = T::max(vmax, T::load(src_ptr_h + kx * pack_c));
                }
            }

            T::save(dst_ptr, vmax);
        }
    }
}

/*
max pooling 3x3s2 kernel
*/
template <class T, int pack_c>
void X86MaxPoolingCenter3x3s2(const float* src, long iw, long ih, float* dst, long ow, long oh, long pad_w, long pad_h, long l,
                           long r, long t, long b) {
    for (long oy = t; oy < b; ++oy) {
        for (long ox = l; ox < r; ++ox) {
            T vmax(-FLT_MAX);

            const long src_offset_x = ox * 2 - pad_w;
            const long src_offset_y = oy * 2 - pad_h;
            const auto src_ptr      = src + (src_offset_y * iw + src_offset_x) * pack_c;
            auto dst_ptr            = dst + (oy * ow + ox) * pack_c;

            for (long ky = 0; ky < 3; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * pack_c;
                vmax                 = T::max(vmax, T::load(src_ptr_h + 0 * pack_c));
                vmax                 = T::max(vmax, T::load(src_ptr_h + 1 * pack_c));
                vmax                 = T::max(vmax, T::load(src_ptr_h + 2 * pack_c));
            }
            T::save(dst_ptr, vmax);
        }
    }
}

/*
general max pooling center kernel
*/
template <class T, int pack_c>
void X86MaxPoolingCenter(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh, long stride_w,
                      long stride_h, long pad_w, long pad_h, long l, long r, long t, long b) {
    for (long oy = t; oy < b; ++oy) {
        for (long ox = l; ox < r; ++ox) {
            T vmax(-FLT_MAX);

            const long src_offset_x = ox * stride_w - pad_w;
            const long src_offset_y = oy * stride_h - pad_h;
            const auto src_ptr      = src + (src_offset_y * iw + src_offset_x) * pack_c;
            auto dst_ptr            = dst + (oy * ow + ox) * pack_c;

            for (long ky = 0; ky < kh; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * pack_c;
                for (long kx = 0; kx < kw; kx++) {
                    vmax = T::max(vmax, T::load(src_ptr_h + kx * pack_c));
                }
            }

            T::save(dst_ptr, vmax);
        }
    }
}

/*
max pooling func, process four corners and center
*/
template <class T, int pack_c>
void X86MaxPooling(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h, long l, long r, long t, long b) {
    // top corner
    X86MaxPoolingCorner<T, pack_c>(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, 0, ow, 0, t);
    if (kw == 3 && kh == 3 && stride_h == 2 && stride_w == 2) {
        X86MaxPoolingCenter3x3s2<T, pack_c>(src, iw, ih, dst, ow, oh, pad_w, pad_h, l, r, t, b);
    } else {
        X86MaxPoolingCenter<T, pack_c>(src, iw, ih, dst, ow, oh, kw, kh, stride_w, stride_h, pad_w, pad_h, l, r, t, b);
    }

    // bottom corner
    X86MaxPoolingCorner<T, pack_c>(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, 0, ow, b, oh);
    // left corner
    X86MaxPoolingCorner<T, pack_c>(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, 0, l, t, b);
    // right corner
    X86MaxPoolingCorner<T, pack_c>(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, r, ow, t, b);
}

template void X86MaxPooling<Float8, 8>(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h, long l, long r, long t, long b);
template void X86MaxPooling<Float4, 4>(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h, long l, long r, long t, long b);

/*
general avg pooling func
*/
template <class T, int pack_c>
void X86AvgPooling(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h) {
    for (long oy = 0; oy < oh; ++oy) {
        for (long ox = 0; ox < ow; ++ox) {
            T vavg(0.f);

            const long srcOriginX    = ox * stride_w - pad_w;
            const long srcOriginY    = oy * stride_h - pad_h;
            const long kxs           = MAX(0, -srcOriginX);
            const long kxe           = MIN(kw, iw - srcOriginX);
            const long kys           = MAX(0, -srcOriginY);
            const long kye           = MIN(kh, ih - srcOriginY);
            const float kernel_count = 1.0 / ((kxe - kxs) * (kye - kys));
            const auto src_ptr       = src + (srcOriginY * iw + srcOriginX) * pack_c;
            auto dst_ptr             = dst + (oy * ow + ox) * pack_c;

            for (long ky = kys; ky < kye; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * pack_c;
                for (long kx = kxs; kx < kxe; kx++) {
                    vavg = vavg + T::load(src_ptr_h + kx * pack_c);
                }
            }

            vavg = vavg * T(kernel_count);
            T::save(dst_ptr, vavg);
        }
    }
}

template void X86AvgPooling<Float8, 8>(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h);
template void X86AvgPooling<Float4, 4>(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h);

template <class T, int pack_c>
Status X86_FMA(float *input_data, float *output_data, float *scale_data, float *bias_data,
               bool shared_channel, bool has_bias, DimsVector output_dim) {
    
    int channel = output_dim[1];
    int cal_count;
    if (shared_channel)
        cal_count = DimsVectorUtils::Count(output_dim);
    else
        cal_count = DimsVectorUtils::Count(output_dim, 2);
    
    if (shared_channel) {
        int tail = cal_count - cal_count % pack_c;
        if (has_bias) {
            T src;
            T scale = T(&scale_data[0]);
            T bias  = T(&bias_data[0]);
            for (size_t i = 0; i < tail; i += pack_c) {
                src = T::loadu(input_data + i);
                T::mla_123(src, scale, bias);
                T::saveu(output_data + i, src);
            }
            for (size_t i = tail; i < cal_count; i++) {
                output_data[i] = input_data[i] * scale_data[0] + bias_data[0];
            }
        } else { // no bias
            T src;
            T scale = T(&scale_data[0]);
            for (size_t i = 0; i < tail; i += pack_c) {
                src = T::loadu(input_data + i);
                src = T::mul(src, scale);
                T::saveu(output_data + i, src);
            }
            for (size_t i = tail; i < cal_count; i++) {
                output_data[i] = input_data[i] * scale_data[0];
            }
        }
    } else {
        int tail = cal_count - cal_count % pack_c;
        for (int b = 0; b < output_dim[0]; b++) {
            for (int c = 0; c < channel; c++) {
                if (has_bias) {
                    // register __m256 src, scale, bias;
                    T src;
                    T scale = T(&scale_data[c]);
                    T bias  = T(&bias_data[c]);
                    float *input  = input_data + (b * channel + c) * cal_count;
                    float *output = output_data + (b * channel + c) * cal_count;
                    for (size_t index = 0; index < tail; index += pack_c) {
                        src = T::loadu(input + index);
                        T::mla_123(src, scale, bias);
                        T::saveu(output + index, src);
                    }
                    for (size_t index = tail; index < cal_count; index++) {
                        output[index] = input[index] * scale_data[c] + bias_data[c];
                    }
                } else {
                    // register __m256 src, scale;
                    T src;
                    T scale = T(&scale_data[c]);
                    float *input  = input_data + (b * channel + c) * cal_count;
                    float *output = output_data + (b * channel + c) * cal_count;
                    for (size_t index = 0; index < tail; index += pack_c) {
                        src = T::loadu(input + index);
                        src = T::mul(src, scale);
                        T::saveu(output + index, src);
                    }
                    for (size_t index = tail; index < cal_count; index++)
                        output[index] = input[index] * scale_data[c];
                }
            }
        }
    }
    return TNN_OK;
}
template Status X86_FMA<Float8, 8>(float *input_data, float *output_data, float *scale_data, float *bias_data,
               bool shared_channel, bool has_bias, DimsVector output_dim);
template Status X86_FMA<Float4, 4>(float *input_data, float *output_data, float *scale_data, float *bias_data,
               bool shared_channel, bool has_bias, DimsVector output_dim);

template<class T, int pack>
Status X86_GroupNorm_FMA(
    float *input_data, float *output_data,
    float *scale_data, float *bias_data,
    int group, float epsilon,
    int batch_time_group, int channels_per_group, int channel_area, int group_area)
{
    const int tail_channel_area = channel_area - channel_area % pack;
    const int tail_group_area   = group_area   - group_area   % pack;

    for (int b = 0; b < batch_time_group; b++) {
        double mean_x = 0;
        double variance = 1;
        
        // get mean and variance
        {
            T sum_x_pack(0.f);
            T sum_x2_pack(0.f);
            T src;
            for (int hw = 0; hw < tail_group_area; hw += pack) {
                src         = T::loadu(input_data + hw);
                sum_x_pack  = T::add(src, sum_x_pack);
                T::mla(sum_x2_pack, src, src);
            }
            double sum_x = 0., sum_x2 = 0.;
            // for (int i = 0; i < pack; i++) {
            //     sum_x   += sum_x_pack.value[i];
            //     sum_x2  += sum_x2_pack.value[i];
            // }
            sum_x  = T::reduce_add(sum_x_pack);
            sum_x2 = T::reduce_add(sum_x2_pack);
            for (int hw = tail_group_area; hw < group_area; hw++) {
                sum_x   += input_data[hw];
                sum_x2  += input_data[hw] * input_data[hw];
            }
            mean_x  = sum_x / group_area;
            auto mean_x2 = sum_x2 / group_area;

            variance = mean_x2 - mean_x * mean_x;
            variance = 1.0f / sqrt(variance + epsilon);
        }

        int output_channel = (b % group) * channels_per_group;
        for (int c = 0; c < channels_per_group; ++c, ++output_channel) {
            float k = scale_data[output_channel];
            float bias = bias_data == NULL ? 0.0f : bias_data[output_channel];
            bias -= mean_x * variance * k;
            k = k * variance;
            T k_pack(k);
            T bias_pack(bias);
            T temp;
            for (int hw = 0; hw < tail_channel_area; hw += pack, output_data += pack, input_data += pack) {
                temp = T::loadu(input_data);
                T::mla_123(temp, k_pack, bias_pack);
                T::saveu(output_data, temp);
            }
            for (int hw = tail_channel_area; hw < channel_area; hw++, output_data++, input_data++) {
                *output_data = (float)((*input_data) * k + bias);
            }
        }
    }

    return TNN_OK;
}

template Status X86_GroupNorm_FMA<Float4, 4>(float *input_data, float *output_data,
    float *scale_data, float *bias_data,
    int group, float epsilon,
    int batch_time_group, int channels_per_group, int channel_area, int group_area);

template Status X86_GroupNorm_FMA<Float8, 8>(float *input_data, float *output_data,
    float *scale_data, float *bias_data,
    int group, float epsilon,
    int batch_time_group, int channels_per_group, int channel_area, int group_area);

template<X86ReduceOpType type>
float reduce_iter_op(const float acc, const float v) {
    return acc + v;
}

template<X86ReduceOpType type>
float reduce_final_op(const float acc, const float num) {
    return acc;
}

template<> float reduce_iter_op<X86ReduceOpType::kL1>(const float acc, const float v) {return acc + std::abs(v); }
template<> float reduce_iter_op<X86ReduceOpType::kMIN>(const float acc, const float v) {return std::min(acc, v); }
template<> float reduce_iter_op<X86ReduceOpType::kMAX>(const float acc, const float v) {return std::max(acc, v); }
template<> float reduce_iter_op<X86ReduceOpType::kPROD>(const float acc, const float v) {return acc * v; }

template<> float reduce_final_op<X86ReduceOpType::kMEAN>(const float acc, const float num) {return acc / num; }

template<X86ReduceOpType type>
void reduce_preprocess(float* input, float*& output, float* workspace, size_t count) {
    output = input;
}

template<X86ReduceOpType type>
void reduce_postprocess(float* input, float* output, size_t count) {
    memcpy(output, input, count * sizeof(float));
}

template<> void reduce_preprocess<X86ReduceOpType::kL2>(float* input, float*& output, float* workspace, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        workspace[i] = std::pow(input[i], 2);
    }
    output = workspace;
}

template<> void reduce_postprocess<X86ReduceOpType::kL2>(float* input, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = std::sqrt(input[i]);
    }
}

template<> void reduce_preprocess<X86ReduceOpType::kLOGSUMEXP>(float* input, float*& output, float* workspace, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        workspace[i] = std::exp(input[i]);
    }
    output = workspace;
}

template<> void reduce_postprocess<X86ReduceOpType::kLOGSUMEXP>(float* input, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = std::log(input[i]);
    }
}

template<> void reduce_preprocess<X86ReduceOpType::kSUMSQUARE>(float* input, float*& output, float* workspace, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        workspace[i] = std::pow(input[i], 2);
    }
    output = workspace;
}

template<> void reduce_postprocess<X86ReduceOpType::kLOGSUM>(float* input, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = std::log(input[i]);
    }
}

template<X86ReduceOpType type>
void reduce_kernel(float * input, float * output, size_t outer_size, size_t inner_size, size_t reduce_size) 
{
    for(long outer_idx = 0; outer_idx < outer_size; outer_idx++) {
        OMP_PARALLEL_FOR_GUIDED_
        for(long inner_idx = 0; inner_idx < inner_size; inner_idx++) {
            float acc = 0;
            if (type == X86ReduceOpType::kMIN) {
                acc = FLT_MAX;
            } else if (type == X86ReduceOpType::kMAX) {
                acc = -FLT_MAX;
            }
            for(int i = 0; i < reduce_size; i++) {
                acc = reduce_iter_op<type>(acc, input[i * inner_size + inner_idx]);
            }
            output[inner_idx] = reduce_final_op<type>(acc, float(reduce_size));
        }
        input += reduce_size * inner_size;
        output += inner_size;
    }
}

using reduce_kernel_ptr_t = decltype(&reduce_kernel<X86ReduceOpType::kMEAN>);
using reduce_preprocess_ptr_t = decltype(&reduce_preprocess<X86ReduceOpType::kMEAN>);
using reduce_postprocess_ptr_t = decltype(&reduce_postprocess<X86ReduceOpType::kMEAN>);

Status X86_REDUCE_CALCULATE(float *input, float *output, float *workspace,
                            std::vector<std::tuple<int, int, int>> &reduce_dims,
                            DimsVector input_dim, DimsVector output_dim, X86ReduceOpType op_type)
{
    reduce_kernel_ptr_t reduce_kernel_ptr = nullptr;
    reduce_preprocess_ptr_t reduce_preprocess_ptr = nullptr;
    reduce_postprocess_ptr_t reduce_postprocess_ptr = nullptr;
    switch (op_type) {
        case X86ReduceOpType::kMEAN:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kMEAN>;
            reduce_preprocess_ptr = reduce_preprocess<X86ReduceOpType::kMEAN>;
            reduce_postprocess_ptr = reduce_postprocess<X86ReduceOpType::kMEAN>;
            break;
        case X86ReduceOpType::kL1:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kL1>;
            reduce_preprocess_ptr = reduce_preprocess<X86ReduceOpType::kL1>;
            reduce_postprocess_ptr = reduce_postprocess<X86ReduceOpType::kL1>;
            break;
        case X86ReduceOpType::kL2:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kL2>;
            reduce_preprocess_ptr = reduce_preprocess<X86ReduceOpType::kL2>;
            reduce_postprocess_ptr = reduce_postprocess<X86ReduceOpType::kL2>;
            break;
        case X86ReduceOpType::kMIN:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kMIN>;
            reduce_preprocess_ptr = reduce_preprocess<X86ReduceOpType::kMIN>;
            reduce_postprocess_ptr = reduce_postprocess<X86ReduceOpType::kMIN>;
            break;
        case X86ReduceOpType::kMAX:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kMAX>;
            reduce_preprocess_ptr = reduce_preprocess<X86ReduceOpType::kMAX>;
            reduce_postprocess_ptr = reduce_postprocess<X86ReduceOpType::kMAX>;
            break;
        case X86ReduceOpType::kSUM:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kSUM>;
            reduce_preprocess_ptr = reduce_preprocess<X86ReduceOpType::kSUM>;
            reduce_postprocess_ptr = reduce_postprocess<X86ReduceOpType::kSUM>;
            break;
        case X86ReduceOpType::kPROD:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kPROD>;
            reduce_preprocess_ptr = reduce_preprocess<X86ReduceOpType::kPROD>;
            reduce_postprocess_ptr = reduce_postprocess<X86ReduceOpType::kPROD>;
            break;
        case X86ReduceOpType::kLOGSUM:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kLOGSUM>;
            reduce_preprocess_ptr = reduce_preprocess<X86ReduceOpType::kLOGSUM>;
            reduce_postprocess_ptr = reduce_postprocess<X86ReduceOpType::kLOGSUM>;
            break;
        case X86ReduceOpType::kLOGSUMEXP:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kLOGSUMEXP>;
            reduce_preprocess_ptr = reduce_preprocess<X86ReduceOpType::kLOGSUMEXP>;
            reduce_postprocess_ptr = reduce_postprocess<X86ReduceOpType::kLOGSUMEXP>;
            break;
        case X86ReduceOpType::kSUMSQUARE:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kSUMSQUARE>;
            reduce_preprocess_ptr = reduce_preprocess<X86ReduceOpType::kSUMSQUARE>;
            reduce_postprocess_ptr = reduce_postprocess<X86ReduceOpType::kSUMSQUARE>;
            break;
        default:
            LOGE("Error, unknown binary op_type\n");
            return TNNERR_LAYER_ERR;
    }

    size_t input_count  = DimsVectorUtils::Count(input_dim);
    size_t output_count = DimsVectorUtils::Count(output_dim);

    float *ping_buf = nullptr;
    float *pong_buf = workspace + input_count;
    reduce_preprocess_ptr(input, ping_buf, workspace, input_count);
    int first = 1;
    for (int i = 0; i < reduce_dims.size(); ++i) {
        auto reduce_dim   = reduce_dims[i];
        auto outer_count  = std::get<0>(reduce_dim);
        auto reduce_count = std::get<1>(reduce_dim);
        auto inner_count  = std::get<2>(reduce_dim);

        reduce_kernel_ptr(ping_buf, pong_buf, outer_count, inner_count, reduce_count);
        if (first) {
            first = 0;
            ping_buf = workspace + input_count;
            pong_buf = workspace;
        } else {
            float *tmp = pong_buf;
            pong_buf = ping_buf;
            ping_buf = tmp;
        }
    }
    reduce_postprocess_ptr(ping_buf, output, output_count);
    return TNN_OK;
}

Status X86_NORMALIZE_CALCULATE(float *input, float *output, int axis,
                               DimsVector input_dim, DimsVector output_dim, int mode, float epsilon) {
    int outer_size = DimsVectorUtils::Count(input_dim, 0, axis);
    int reduce_size = input_dim[axis];
    int inner_size = DimsVectorUtils::Count(input_dim, axis + 1);

    float* tmp = (float*)malloc(outer_size * inner_size * sizeof(float));
    if (mode == 2) { // L2
        reduce_kernel<X86ReduceOpType::kL2>(input, tmp, outer_size, inner_size, reduce_size);
    } else if (mode == 1) { // L1
        reduce_kernel<X86ReduceOpType::kL1>(input, tmp, outer_size, inner_size, reduce_size);
    } else if (mode == INT_MAX) { // MAX
        reduce_kernel<X86ReduceOpType::kMAX>(input, tmp, outer_size, inner_size, reduce_size);
    } else if (mode == INT_MIN) { // MIN
        reduce_kernel<X86ReduceOpType::kMIN>(input, tmp, outer_size, inner_size, reduce_size);
    }

    for (int o = 0; o < outer_size; o++) {
        for (int i = 0; i < inner_size; i++) {
            tmp[o * inner_size + i] = std::max(epsilon, tmp[o * inner_size + i]);
            for (int r = 0; r < reduce_size; r++) {
                output[o * inner_size * reduce_size + r * inner_size + i] = \
                input[o * inner_size * reduce_size + r * inner_size + i] / tmp[o * inner_size + i];
            }
        }
    }

    if(tmp) free(tmp);
    return TNN_OK;
}

template <int activation_type, typename VEC, int pack>
void DepthwiseConv(float* dst, const float* src, const float* weight, const float* bias, long width, long src_w_step, long fw, long fh,
                   long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep) {
    long dx, fx, fy;
    VEC bias_v = VEC::loadu(bias);
    VEC v_zero = VEC(0.f);
    VEC v_6    = VEC(6.f);
    for (long y = 0; y < height; ++y) {
        auto srcY = src + y * srcHStep;
        auto dstY = dst + y * dstHStep;
        dx        = 0;
        for (; dx + 3 < width; dx += 4) {
            VEC dst_v[4];
            for (long i = 0; i < 4; i++)
                dst_v[i] = bias_v;
            const auto* src_z    = srcY + src_w_step * dx;
            const auto* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const auto* src_y    = src_z + fy * dilate_y_step;
                const auto* weight_y = weight_z + fy * fw * pack;
                for (fx = 0; fx < fw; ++fx) {
                    VEC weight_v = VEC::loadu(weight_y + pack * fx);
                    VEC src_v0   = VEC::load(src_y + fx * dilate_x_step);
                    VEC src_v1   = VEC::load(src_y + fx * dilate_x_step + src_w_step);
                    VEC src_v2   = VEC::load(src_y + fx * dilate_x_step + 2 * src_w_step);
                    VEC src_v3   = VEC::load(src_y + fx * dilate_x_step + 3 * src_w_step);
                    VEC::mla(dst_v[0], src_v0, weight_v);
                    VEC::mla(dst_v[1], src_v1, weight_v);
                    VEC::mla(dst_v[2], src_v2, weight_v);
                    VEC::mla(dst_v[3], src_v3, weight_v);
                }
            }
            if (activation_type == ActivationType_ReLU || 
                activation_type == ActivationType_ReLU6) {
                dst_v[0] = VEC::max(dst_v[0], v_zero);
                dst_v[1] = VEC::max(dst_v[1], v_zero);
                dst_v[2] = VEC::max(dst_v[2], v_zero);
                dst_v[3] = VEC::max(dst_v[3], v_zero);
            }
            if (activation_type == ActivationType_ReLU6) {
                dst_v[0] = VEC::min(dst_v[0], v_6);
                dst_v[1] = VEC::min(dst_v[1], v_6);
                dst_v[2] = VEC::min(dst_v[2], v_6);
                dst_v[3] = VEC::min(dst_v[3], v_6);
            }
            VEC::save(dstY + (dx + 0) * pack, dst_v[0]);
            VEC::save(dstY + (dx + 1) * pack, dst_v[1]);
            VEC::save(dstY + (dx + 2) * pack, dst_v[2]);
            VEC::save(dstY + (dx + 3) * pack, dst_v[3]);
        }
        for (; dx < width; ++dx) {
            VEC dst_v = bias_v;
            const auto* src_z    = srcY + src_w_step * dx;
            const auto* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const auto* src_y    = src_z + fy * dilate_y_step;
                const auto* weight_y = weight_z + fy * fw * pack;
                for (fx = 0; fx < fw; ++fx) {
                    VEC src_v    = VEC::load(src_y + fx * dilate_x_step);
                    VEC weight_v = VEC::loadu(weight_y + pack * fx);
                    VEC::mla(dst_v, src_v, weight_v);
                }
            }
            if (activation_type == ActivationType_ReLU || 
                activation_type == ActivationType_ReLU6) {
                dst_v = VEC::max(dst_v, v_zero);
            }
            if (activation_type == ActivationType_ReLU6) {
                dst_v = VEC::min(dst_v, v_6);
            }
            VEC::save(dstY + dx * pack, dst_v);
        }
    }
}

template void DepthwiseConv<ActivationType_None, Float4, 4>(
    float* dst, const float* src, const float* weight, const float* bias, long width, long src_w_step, long fw, long fh,
    long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep);

template void DepthwiseConv<ActivationType_ReLU, Float4, 4>(
    float* dst, const float* src, const float* weight, const float* bias, long width, long src_w_step, long fw, long fh,
    long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep);

template void DepthwiseConv<ActivationType_ReLU6, Float4, 4>(
    float* dst, const float* src, const float* weight, const float* bias, long width, long src_w_step, long fw, long fh,
    long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep);

template void DepthwiseConv<ActivationType_None, Float8, 8>(
    float* dst, const float* src, const float* weight, const float* bias, long width, long src_w_step, long fw, long fh,
    long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep);

template void DepthwiseConv<ActivationType_ReLU, Float8, 8>(
    float* dst, const float* src, const float* weight, const float* bias, long width, long src_w_step, long fw, long fh,
    long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep);

template void DepthwiseConv<ActivationType_ReLU6, Float8, 8>(
    float* dst, const float* src, const float* weight, const float* bias, long width, long src_w_step, long fw, long fh,
    long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep);

template <int left, int oc_>
void X86SgemvLeft(float* dst, const float* src, const float* weight, float *bias, size_t batch_stride) {
    float acc[8];
    for (int i = 0; i < left; i++) {
        acc[i] = bias[i];
    }
    for (size_t ic = 0; ic < batch_stride; ic++) {
        auto weight_ic = weight + ic * oc_;
        for (int i = 0; i < left; i++) {
            acc[i] += weight_ic[i] * src[ic];
        }
    }
    for (int i = 0; i < left; i++) {
        dst[i] = acc[i];
    }
}

template <typename VEC, int pack>
void X86Sgemv(float* dst, const float* src, const float* weight, float *bias, DimsVector dims_input, DimsVector dims_output) {
    size_t batch_stride = DimsVectorUtils::Count(dims_input, 1);
    int oc_vec_size = dims_output[1] / pack * pack;
    int oc_left = dims_output[1] - oc_vec_size;
    for (int b = 0; b < dims_output[0]; ++b) {
        const float *src_batch = src + b * batch_stride;
        float *dst_batch = dst + b * dims_output[1];

        OMP_PARALLEL_FOR_GUIDED_
        for (int oc = 0; oc < oc_vec_size; oc += pack) {
            auto weight_oc = weight + oc * batch_stride;
            VEC acc = VEC::loadu(bias + oc);
            size_t ic = 0;
            for (; ic + 3 < batch_stride; ic += 4) {
                auto weight_ic   = weight_oc + ic * pack;
                VEC src_v0    = VEC(src_batch[ic]);
                VEC src_v1    = VEC(src_batch[ic + 1]);
                VEC src_v2    = VEC(src_batch[ic + 2]);
                VEC src_v3    = VEC(src_batch[ic + 3]);
                VEC weight_v0 = VEC::load(weight_ic);
                VEC weight_v1 = VEC::load(weight_ic + pack * 1);
                VEC weight_v2 = VEC::load(weight_ic + pack * 2);
                VEC weight_v3 = VEC::load(weight_ic + pack * 3);
                VEC::mla(acc, weight_v0, src_v0);
                VEC::mla(acc, weight_v1, src_v1);
                VEC::mla(acc, weight_v2, src_v2);
                VEC::mla(acc, weight_v3, src_v3);
            }
            for (; ic < batch_stride; ic++) {
                VEC src_v    = VEC(src_batch[ic]);
                VEC weight_v = VEC::load(weight_oc + ic * pack);
                VEC::mla(acc, weight_v, src_v);
            }
            VEC::saveu(dst_batch + oc, acc);
        }
        int left = oc_left;
        int oc = oc_vec_size;
        if (pack == 8) {
            if (left == 7) {
                X86SgemvLeft<7, pack>(dst_batch + oc, src_batch, weight + oc * batch_stride, bias + oc, batch_stride);
            } else if (left == 6) {
                X86SgemvLeft<6, pack>(dst_batch + oc, src_batch, weight + oc * batch_stride, bias + oc, batch_stride);
            } else if (left == 5) {
                X86SgemvLeft<5, pack>(dst_batch + oc, src_batch, weight + oc * batch_stride, bias + oc, batch_stride);
            } else if (left == 4) {
                X86SgemvLeft<4, pack>(dst_batch + oc, src_batch, weight + oc * batch_stride, bias + oc, batch_stride);
            }
        }
        if (left == 3) {
            X86SgemvLeft<3, pack>(dst_batch + oc, src_batch, weight + oc * batch_stride, bias + oc, batch_stride);
        } else if (left == 2) {
            X86SgemvLeft<2, pack>(dst_batch + oc, src_batch, weight + oc * batch_stride, bias + oc, batch_stride);
        } else if (left == 1) {
            X86SgemvLeft<1, pack>(dst_batch + oc, src_batch, weight + oc * batch_stride, bias + oc, batch_stride);
        }
    }
}
template void X86Sgemv<Float4, 4>(float* dst, const float* src, const float* weight, float *bias, DimsVector dims_input, DimsVector dims_output);
template void X86Sgemv<Float8, 8>(float* dst, const float* src, const float* weight, float *bias, DimsVector dims_input, DimsVector dims_output);

template <int activation_type, typename VEC, int pack>
void X86_Post_Exec(float *dst, const float *bias, long channel, long area) {
    for (long c = 0; c < channel; c++) {
        auto dst_c = dst + c * area;
        VEC bias_v = VEC(bias + c);
        VEC zero_v = VEC(0.f);
        VEC six_v = VEC(6.f);
        long i = 0;
        for (; i + pack - 1 < area; i += pack) {
            VEC src_v = VEC::loadu(dst_c + i);
            VEC dst_v = VEC::add(src_v, bias_v);

            if (activation_type == ActivationType_ReLU || 
                activation_type == ActivationType_ReLU6) {
                dst_v = VEC::max(dst_v, zero_v);
            }
            if (activation_type == ActivationType_ReLU6) {
                dst_v = VEC::min(dst_v, six_v);
            }
            VEC::saveu(dst_c + i, dst_v);
        }

        for (; i < area; i++) {
            float dst_value = dst_c[i] + bias[c];
            if (activation_type == ActivationType_ReLU || 
                activation_type == ActivationType_ReLU6) {
                dst_value = std::max(dst_value, 0.f);
            }
            if (activation_type == ActivationType_ReLU6) {
                dst_value = std::min(dst_value, 6.f);
            }
            dst_c[i] = dst_value;
        }
    }
}
template void X86_Post_Exec<ActivationType_None, Float4, 4>(float *dst, const float *bias, long channel, long area);
template void X86_Post_Exec<ActivationType_ReLU, Float4, 4>(float *dst, const float *bias, long channel, long area);
template void X86_Post_Exec<ActivationType_ReLU6, Float4, 4>(float *dst, const float *bias, long channel, long area);
template void X86_Post_Exec<ActivationType_None, Float8, 8>(float *dst, const float *bias, long channel, long area);
template void X86_Post_Exec<ActivationType_ReLU, Float8, 8>(float *dst, const float *bias, long channel, long area);
template void X86_Post_Exec<ActivationType_ReLU6, Float8, 8>(float *dst, const float *bias, long channel, long area);

//ActivationType_SIGMOID_MUL TBD

template <typename VEC, int pack>
void X86_VectorAdd(float *dst, const float *src_a, const float *src_b, long len) {
    long i = 0;
    for (; i + pack - 1 < len; i += pack) {
        VEC a_vec = VEC::loadu(src_a + i);
        VEC b_vec = VEC::loadu(src_b + i);
        VEC c_vec = VEC::add(a_vec, b_vec);
        VEC::saveu(dst + i, c_vec);
    }
    for (; i < len; i++) {
        dst[i] = src_a[i] + src_b[i];
    }
}
template void X86_VectorAdd<Float4, 4>(float *dst, const float *src_a, const float *src_b, long len);
template void X86_VectorAdd<Float8, 8>(float *dst, const float *src_a, const float *src_b, long len);

template <typename VEC, int pack>
void X86_VectorAdd(float *dst, const float *src, long len) {
    long i = 0;
    for (; i + pack - 1 < len; i += pack) {
        VEC a_vec = VEC::loadu(src + i);
        VEC b_vec = VEC::loadu(dst + i);
        VEC c_vec = VEC::add(a_vec, b_vec);
        VEC::saveu(dst + i, c_vec);
    }
    for (; i < len; i++) {
        dst[i] += src[i];
    }
}
template void X86_VectorAdd<Float4, 4>(float *dst, const float *src, long len);
template void X86_VectorAdd<Float8, 8>(float *dst, const float *src, long len);

void X86StrideSliceImpl(DimsVector begins, DimsVector strides, DimsVector dims_output,
                        DimsVector input_strides, DimsVector output_strides,
                        const float* input_data, float* output_data) {
    if (dims_output.size() == 5) {
        for (int n = 0, n_idx = begins[0]; n < dims_output[0]; n++, n_idx += strides[0]) {
            auto input_n = input_data + n_idx * input_strides[0];
            auto output_n = output_data + n * output_strides[0];
            for (int c = 0, c_idx = begins[1]; c < dims_output[1]; c++, c_idx += strides[1]) {
                auto input_c = input_n + c_idx * input_strides[1];
                auto output_c = output_n + c * output_strides[1];
                for (int h = 0, h_idx = begins[2]; h < dims_output[2]; h++, h_idx += strides[2]) {
                    auto input_h = input_c + h_idx * input_strides[2];
                    auto output_h = output_c + h * output_strides[2];
                    for (int w = 0, w_idx = begins[3]; w < dims_output[3]; w++, w_idx += strides[3]) {
                        auto input_w = input_h + w_idx * input_strides[3];
                        auto output_w = output_h + w * output_strides[3];
                        for (int x = 0, x_idx = begins[4]; x < dims_output[4]; x++, x_idx += strides[4]) {
                            output_w[x] = input_w[x_idx];
                        }
                    }
                }
            }
        }
    } else if (dims_output.size() == 4) {
        for (int n = 0, n_idx = begins[0]; n < dims_output[0]; n++, n_idx += strides[0]) {
            auto input_n = input_data + n_idx * input_strides[0];
            auto output_n = output_data + n * output_strides[0];
            for (int c = 0, c_idx = begins[1]; c < dims_output[1]; c++, c_idx += strides[1]) {
                auto input_c = input_n + c_idx * input_strides[1];
                auto output_c = output_n + c * output_strides[1];
                for (int h = 0, h_idx = begins[2]; h < dims_output[2]; h++, h_idx += strides[2]) {
                    auto input_h = input_c + h_idx * input_strides[2];
                    auto output_h = output_c + h * output_strides[2];
                    for (int w = 0, w_idx = begins[3]; w < dims_output[3]; w++, w_idx += strides[3]) {
                        output_h[w] = input_h[w_idx];
                    }
                }
            }
        }
    } else if (dims_output.size() == 3) {
        for (int n = 0, n_idx = begins[0]; n < dims_output[0]; n++, n_idx += strides[0]) {
            auto input_n = input_data + n_idx * input_strides[0];
            auto output_n = output_data + n * output_strides[0];
            for (int c = 0, c_idx = begins[1]; c < dims_output[1]; c++, c_idx += strides[1]) {
                auto input_c = input_n + c_idx * input_strides[1];
                auto output_c = output_n + c * output_strides[1];
                for (int h = 0, h_idx = begins[2]; h < dims_output[2]; h++, h_idx += strides[2]) {
                    output_c[h] = input_c[h_idx];
                }
            }
        }
    } else if (dims_output.size() == 2) {
        for (int n = 0, n_idx = begins[0]; n < dims_output[0]; n++, n_idx += strides[0]) {
            auto input_n = input_data + n_idx * input_strides[0];
            auto output_n = output_data + n * output_strides[0];
            for (int c = 0, c_idx = begins[1]; c < dims_output[1]; c++, c_idx += strides[1]) {
                output_n[c] = input_n[c_idx];
            }
        }
    } else if (dims_output.size() == 1) {
        for (int n = 0, n_idx = begins[0]; n < dims_output[0]; n++, n_idx += strides[0]) {
            output_data[n] = input_data[n_idx];
        }
    }
}

}
