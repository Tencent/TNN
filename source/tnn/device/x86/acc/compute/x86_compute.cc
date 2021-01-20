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

#include "x86_compute.h"
#include "tnn/device/x86/acc/Float8.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <type_traits>
#include <immintrin.h>

#include "jit/cblas.h"

namespace TNN_NS {

static std::vector<int> dims_to_steps(std::vector<int> dims) {
    std::vector<int> ret(dims.size(), 1);
    int cnt = 1;
    for(int i=dims.size() - 1;i>=0;i--) {
        if (dims[i] == 1) {
            ret[i] = 0;
        } else {
            ret[i] = cnt;
            cnt *= dims[i];
        }
    }
    return ret;
}

template<X86BinaryOpType type>
float binary_op(const float a, const float b) {
    return a;
}

template<> float binary_op<X86BinaryOpType::kADD>(const float a, const float b) {
    return a + b;
}

template<> float binary_op<X86BinaryOpType::kSUB>(const float a, const float b) {
    return a - b;
}

template<> float binary_op<X86BinaryOpType::kMUL>(const float a, const float b) {
    return a * b;
}

template<> float binary_op<X86BinaryOpType::kDIV>(const float a, const float b) {
    return a / b;
}


template<X86BinaryOpType type>
void binary_kernel(std::vector<int> output_dims, const float * a, std::vector<int> steps_a, const float * b, std::vector<int> steps_b, 
                    float * c, std::vector<int> steps_c) 
{

    size_t idx_a = 0;
    size_t idx_b = 0;
    size_t idx_c = 0;

    const int MAX_DIM = 5;

    int d[MAX_DIM] = {1, 1, 1 ,1 ,1};
    int step_a[MAX_DIM] = {0, 0, 0, 0, 0};
    int step_b[MAX_DIM] = {0, 0, 0, 0, 0};
    int step_c[MAX_DIM] = {0, 0, 0, 0, 0};

    int offset = MAX_DIM - output_dims.size();
    memcpy(d + offset, &output_dims[0], output_dims.size() * sizeof(float));
    memcpy(step_a + offset, &steps_a[0], steps_a.size() * sizeof(float));
    memcpy(step_b + offset, &steps_b[0], steps_b.size() * sizeof(float));
    memcpy(step_c + offset, &steps_c[0], steps_c.size() * sizeof(float));

    for(int d0=0;d0<d[0];d0++) {
        for(int d1=0;d1<d[1];d1++) {
            for(int d2=0;d2<d[2];d2++) {
                for(int d3=0;d3<d[3];d3++) {
                    for(int d4=0;d4<d[4];d4++) {
                        c[idx_c] = binary_op<type>(a[idx_a], b[idx_b]);
                        idx_a += step_a[4];
                        idx_b += step_b[4];
                        idx_c += step_c[4];
                    }
                    idx_a += (step_a[3] - d[4] * step_a[4]);
                    idx_b += (step_b[3] - d[4] * step_b[4]);
                    idx_c += (step_c[3] - d[4] * step_c[4]);
                }
                idx_a += (step_a[2] - d[3] * step_a[3]);
                idx_b += (step_b[2] - d[3] * step_b[3]);
                idx_c += (step_c[2] - d[3] * step_c[3]);
            }
            idx_a += (step_a[1] - d[2] * step_a[2]);
            idx_b += (step_b[1] - d[2] * step_b[2]);
            idx_c += (step_c[1] - d[2] * step_c[2]);
        }
        idx_a += (step_a[0] - d[1] * step_a[1]);
        idx_b += (step_b[0] - d[1] * step_b[1]);
        idx_c += (step_c[0] - d[1] * step_c[1]);
    }


}

using binary_kernel_func_t = decltype(&binary_kernel<X86BinaryOpType::kADD>);

Status X86_BINARY_CALCULATE(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes, 
                            void *output, DimsVector output_shape,  X86BinaryOpType op_type) {


    if (input_shapes[0].size() != input_shapes[1].size() || 
        input_shapes[0].size() != output_shape.size()) {
        LOGE("Error, shape len not equal\n");
        return TNNERR_LAYER_ERR;
    }

    std::vector<int> steps_a = dims_to_steps(input_shapes[0]);
    std::vector<int> steps_b = dims_to_steps(input_shapes[1]);
    std::vector<int> steps_c = dims_to_steps(output_shape);
    
    binary_kernel_func_t binary_kernel_function = nullptr;

    switch(op_type) {
        case X86BinaryOpType::kADD :
            binary_kernel_function = binary_kernel<X86BinaryOpType::kADD>;
            break;
        case X86BinaryOpType::kSUB :
            binary_kernel_function = binary_kernel<X86BinaryOpType::kSUB>;
            break;
        case X86BinaryOpType::kMUL :
            binary_kernel_function = binary_kernel<X86BinaryOpType::kMUL>;
            break;
        case X86BinaryOpType::kDIV :
            binary_kernel_function = binary_kernel<X86BinaryOpType::kDIV>;
            break;
        default :
            LOGE("Error, unknown binary op_type\n");
            return TNNERR_LAYER_ERR;
    }

    binary_kernel_function(output_shape, (const float *)input_ptrs[0], steps_a, (const float *)input_ptrs[1], steps_b, (float *)output, steps_c);

    return TNN_OK; 
}

Status X86_IM2COL(float* src, int channel, int height, int width, int kernelh, int kernelw, 
              int padh, int padw, int strideh, int stridew, int dilationh, int dilationw, float* dst) {
    int height_col = (height + 2 * padh - dilationh * (kernelh - 1) - 1) / strideh + 1;
    int width_col  = (width + 2 * padw - dilationw * (kernelw - 1) - 1) / stridew + 1;
    int channels_col = channel * kernelh * kernelw;

    // im2col
    for (int c = 0; c < channels_col; c++) {
        int w_offset = c % kernelw;
        int h_offset = (c / kernelw) % kernelh;
        int c_im = c / kernelh / kernelw;
        for (int h = 0; h < height_col; h++) {
            for (int w = 0; w < width_col; w++) {
                int h_pad = h * strideh - padh + h_offset * dilationh;
                int w_pad = w * stridew - padw + w_offset * dilationw;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                    dst[(c * height_col + h) * width_col + w] = src[(c_im * height + h_pad) * width + w_pad];
                else
                    dst[(c * height_col + h) * width_col + w] = 0;
            }
        }
    }

    return TNN_OK;
}

Status X86_matrixMul(int m, int n, int k, const float *A, const float *B, float *C,
                     int has_bias, const float *bias, int activation_type) {

    conv_cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                n, m, k, B, n, A, k, C, n, bias, activation_type);
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
void MaxPoolingCornerAVX(const float* src, long iw, long ih, float* dst, long ow, long kw, long kh, long stride_w, long stride_h,
                      long pad_w, long pad_h, long l, long r, long t, long b) {
    for (long oy = t; oy < b; ++oy) {
        for (long ox = l; ox < r; ++ox) {
            Float8 vmax(-FLT_MAX);

            const long srcOriginX = ox * stride_w - pad_w;
            const long srcOriginY = oy * stride_h - pad_h;
            const long kxs        = MAX(0, -srcOriginX);
            const long kxe        = MIN(kw, iw - srcOriginX);
            const long kys        = MAX(0, -srcOriginY);
            const long kye        = MIN(kh, ih - srcOriginY);
            const auto src_ptr    = src + (srcOriginY * iw + srcOriginX) * 8;
            auto dst_ptr          = dst + (oy * ow + ox) * 8;

            for (long ky = kys; ky < kye; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * 8;
                for (long kx = kxs; kx < kxe; kx++) {
                    vmax = Float8::max(vmax, Float8::load(src_ptr_h + kx * 8));
                }
            }

            Float8::save(dst_ptr, vmax);
        }
    }
}

/*
max pooling 3x3s2 kernel
*/
void MaxPoolingCenter3x3s2AVX(const float* src, long iw, long ih, float* dst, long ow, long oh, long pad_w, long pad_h, long l,
                           long r, long t, long b) {
    for (long oy = t; oy < b; ++oy) {
        for (long ox = l; ox < r; ++ox) {
            Float8 vmax(-FLT_MAX);

            const long src_offset_x = ox * 2 - pad_w;
            const long src_offset_y = oy * 2 - pad_h;
            const auto src_ptr      = src + (src_offset_y * iw + src_offset_x) * 8;
            auto dst_ptr            = dst + (oy * ow + ox) * 8;

            for (long ky = 0; ky < 3; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * 8;
                vmax                 = Float8::max(vmax, Float8::load(src_ptr_h + 0 * 8));
                vmax                 = Float8::max(vmax, Float8::load(src_ptr_h + 1 * 8));
                vmax                 = Float8::max(vmax, Float8::load(src_ptr_h + 2 * 8));
            }
            Float8::save(dst_ptr, vmax);
        }
    }
}

/*
general max pooling center kernel
*/
void MaxPoolingCenterAVX(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh, long stride_w,
                      long stride_h, long pad_w, long pad_h, long l, long r, long t, long b) {
    for (long oy = t; oy < b; ++oy) {
        for (long ox = l; ox < r; ++ox) {
            Float8 vmax(-FLT_MAX);

            const long src_offset_x = ox * stride_w - pad_w;
            const long src_offset_y = oy * stride_h - pad_h;
            const auto src_ptr      = src + (src_offset_y * iw + src_offset_x) * 8;
            auto dst_ptr            = dst + (oy * ow + ox) * 8;

            for (long ky = 0; ky < kh; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * 8;
                for (long kx = 0; kx < kw; kx++) {
                    vmax = Float8::max(vmax, Float8::load(src_ptr_h + kx * 8));
                }
            }

            Float8::save(dst_ptr, vmax);
        }
    }
}

/*
max pooling func, process four corners and center
*/
void MaxPoolingAVX(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h, long l, long r, long t, long b) {
    // top corner
    MaxPoolingCornerAVX(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, 0, ow, 0, t);
    if (kw == 3 && kh == 3 && stride_h == 2 && stride_w == 2) {
        MaxPoolingCenter3x3s2AVX(src, iw, ih, dst, ow, oh, pad_w, pad_h, l, r, t, b);
    } else {
        MaxPoolingCenterAVX(src, iw, ih, dst, ow, oh, kw, kh, stride_w, stride_h, pad_w, pad_h, l, r, t, b);
    }

    // bottom corner
    MaxPoolingCornerAVX(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, 0, ow, b, oh);
    // left corner
    MaxPoolingCornerAVX(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, 0, l, t, b);
    // right corner
    MaxPoolingCornerAVX(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, r, ow, t, b);
}

/*
general avg pooling func
*/
void AvgPoolingAVX(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h) {
    for (long oy = 0; oy < oh; ++oy) {
        for (long ox = 0; ox < ow; ++ox) {
            Float8 vavg(0.f);

            const long srcOriginX    = ox * stride_w - pad_w;
            const long srcOriginY    = oy * stride_h - pad_h;
            const long kxs           = MAX(0, -srcOriginX);
            const long kxe           = MIN(kw, iw - srcOriginX);
            const long kys           = MAX(0, -srcOriginY);
            const long kye           = MIN(kh, ih - srcOriginY);
            const float kernel_count = 1.0 / ((kxe - kxs) * (kye - kys));
            const auto src_ptr       = src + (srcOriginY * iw + srcOriginX) * 8;
            auto dst_ptr             = dst + (oy * ow + ox) * 8;

            for (long ky = kys; ky < kye; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * 8;
                for (long kx = kxs; kx < kxe; kx++) {
                    vavg = vavg + Float8::load(src_ptr_h + kx * 8);
                }
            }

            vavg = vavg * Float8(kernel_count);
            Float8::save(dst_ptr, vavg);
        }
    }
}

Status X86_FMA(float *input_data, float *output_data, float *scale_data, float *bias_data,
               bool shared_channel, bool has_bias, DimsVector output_dim) {
    
    int channel = output_dim[1];
    int cal_count;
    if (shared_channel)
        cal_count = DimsVectorUtils::Count(output_dim);
    else
        cal_count = DimsVectorUtils::Count(output_dim, 2);
    
    if (shared_channel) {
#ifdef __AVX2__
        int tail = cal_count - cal_count % 8;
        if (has_bias) {     // has bias
            // register __m256 src, scale, bias;
            __m256 src, scale, bias;
            scale = _mm256_broadcast_ss(&scale_data[0]);
            bias  = _mm256_broadcast_ss(&bias_data[0]);
            for (size_t i = 0; i < tail; i += 8) {
                src = _mm256_loadu_ps(input_data + i);
                src = _mm256_fmadd_ps(src, scale, bias);
                _mm256_storeu_ps(output_data + i, src);
            }
            for (size_t i = tail; i < cal_count; i++) {
                output_data[i] = input_data[i] * scale_data[0] + bias_data[0];
            }
        } else {        // no bias
            // register __m256 src, scale;
            __m256 src, scale;
            scale = _mm256_broadcast_ss(&scale_data[0]);
            for (size_t i = 0; i < tail; i += 8) {
                src = _mm256_loadu_ps(input_data + i);
                src = _mm256_mul_ps(src, scale);
                _mm256_storeu_ps(output_data + i, src);
            }
            for (size_t i = tail; i < cal_count; i++) {
                output_data[i] = input_data[i] * scale_data[0];
            }
        }  
#else
        const float scale = scale_data[0];
        float bias = 0.f;
        if (has_bias) bias = bias_data[0];
        for (int index = 0; index < cal_count; index++) {
            if (has_bias)
                output_data[index] = input_data[index] * scale + bias;
            else 
                output_data[index] = input_data[index] * scale;
        }
#endif
    } else {
#ifdef __AVX2__
        int tail = cal_count - cal_count % 8;
        for (int b = 0; b < output_dim[0]; b++) {
            for (int c = 0; c < channel; c++) {
                if (has_bias) {
                    // register __m256 src, scale, bias;
                    __m256 src, scale, bias;
                    scale = _mm256_broadcast_ss(&scale_data[c]);
                    bias  = _mm256_broadcast_ss(&bias_data[c]);
                    float *input  = input_data + (b * channel + c) * cal_count;
                    float *output = output_data + (b * channel + c) * cal_count;
                    for (size_t index = 0; index < tail; index += 8) {
                        src = _mm256_loadu_ps(input + index);
                        src = _mm256_fmadd_ps(src, scale, bias);
                        _mm256_storeu_ps(output + index, src);
                    }
                    for (size_t index = tail; index < cal_count; index++)
                        output[index] = input[index] * scale_data[c] + bias_data[c];
                } else {
                    // register __m256 src, scale;
                    __m256 src, scale;
                    scale = _mm256_broadcast_ss(&scale_data[c]);
                    float *input  = input_data + (b * channel + c) * cal_count;
                    float *output = output_data + (b * channel + c) * cal_count;
                    for (size_t index = 0; index < tail; index += 8) {
                        src = _mm256_loadu_ps(input + index);
                        src = _mm256_mul_ps(src, scale);
                        _mm256_storeu_ps(output + index, src);
                    }
                    for (size_t index = tail; index < cal_count; index++)
                        output[index] = input[index] * scale_data[c];
                }
            }
        }
#else
        for (int b = 0; b < output_dim[0]; b++) {
            for (int c = 0; c < channel; c++) {
                float *input  = input_data + (b * channel + c) * cal_count;
                float *output = output_data + (b * channel + c) * cal_count;
                const float scale = scale_data[c];
                float bias = 0.f;
                if (has_bias) bias = bias_data[c];
                for (int index = 0; index < cal_count; index++) {
                    if (has_bias)
                        output[index] = input[index] * scale + bias;
                    else
                        output[index] = input[index] * scale;
                }
            }
        }
#endif
    }
    return TNN_OK;
}


template<X86ReduceOpType type>
float reduce_iter_op(const float acc, const float v) {
    return acc;
}

template<X86ReduceOpType type>
float reduce_final_op(const float acc, const float num) {
    return acc;
}

template<> float reduce_iter_op<X86ReduceOpType::kMEAN>(const float acc, const float v) {return acc + v; }
template<> float reduce_iter_op<X86ReduceOpType::kL1>(const float acc, const float v) {return acc + std::abs(v); }
template<> float reduce_iter_op<X86ReduceOpType::kL2>(const float acc, const float v) {return acc + v * v; }
template<> float reduce_iter_op<X86ReduceOpType::kMIN>(const float acc, const float v) {return std::min(acc, v); }
template<> float reduce_iter_op<X86ReduceOpType::kMAX>(const float acc, const float v) {return std::max(acc, v); }
template<> float reduce_iter_op<X86ReduceOpType::kSUM>(const float acc, const float v) {return acc + v; }
template<> float reduce_iter_op<X86ReduceOpType::kPROD>(const float acc, const float v) {return acc * v; }
template<> float reduce_iter_op<X86ReduceOpType::kLOGSUM>(const float acc, const float v) {return acc + v; }
template<> float reduce_iter_op<X86ReduceOpType::kLOGSUMEXP>(const float acc, const float v) {return acc + std::exp(v); }
template<> float reduce_iter_op<X86ReduceOpType::kSUMSQUARE>(const float acc, const float v) {return acc + v * v; }

template<> float reduce_final_op<X86ReduceOpType::kMEAN>(const float acc, const float num) {return acc / num; }
template<> float reduce_final_op<X86ReduceOpType::kL1>(const float acc, const float num) {return acc; }
template<> float reduce_final_op<X86ReduceOpType::kL2>(const float acc, const float num) {return sqrt(acc); }
template<> float reduce_final_op<X86ReduceOpType::kMIN>(const float acc, const float num) {return acc; }
template<> float reduce_final_op<X86ReduceOpType::kMAX>(const float acc, const float num) {return acc; }
template<> float reduce_final_op<X86ReduceOpType::kSUM>(const float acc, const float num) {return acc; }
template<> float reduce_final_op<X86ReduceOpType::kPROD>(const float acc, const float num) {return acc; }
template<> float reduce_final_op<X86ReduceOpType::kLOGSUM>(const float acc, const float num) {return std::log(acc); }
template<> float reduce_final_op<X86ReduceOpType::kLOGSUMEXP>(const float acc, const float num) {return std::log(acc); }
template<> float reduce_final_op<X86ReduceOpType::kSUMSQUARE>(const float acc, const float num) {return acc; }


template<X86ReduceOpType type>
void reduce_kernel(float * input, float * output, size_t outer_size, size_t inner_size, size_t reduce_size) 
{
    for(size_t outer_idx=0;outer_idx<outer_size;outer_idx++) {
        for(size_t inner_idx=0;inner_idx<inner_size;inner_idx++) {
            float acc = 0;
            for(int i=0;i<reduce_size;i++) {
                acc = reduce_iter_op<type>(acc, input[i * inner_size + inner_idx]);
            }
            output[inner_idx] = reduce_final_op<type>(acc, float(reduce_size));
        }
        input += reduce_size * inner_size;
        output += inner_size;
    }
}

using reduce_kernel_ptr_t  = decltype(&reduce_kernel<X86ReduceOpType::kMEAN>);

Status X86_REDUCE_CALCULATE(float *input, float *output, std::vector<int> axes,
                            DimsVector input_dim, DimsVector output_dim, X86ReduceOpType op_type) 
{
    int outer_begin = 0;
    int outer_end = input_dim.size() - 1;
    int inner_begin = 0;
    int inner_end = input_dim.size() - 1;
    for(int axis : axes) {
        inner_begin = std::max(inner_begin, axis);
        outer_end = std::min(outer_end, axis);
    }

    size_t outer_size = DimsVectorUtils::Count(input_dim, outer_begin, outer_end);
    size_t inner_size = DimsVectorUtils::Count(input_dim, inner_begin, inner_end);
    size_t reduce_size = 1;
    for(int axis : axes) {
        reduce_size *= input_dim[axis];
    }

    reduce_kernel_ptr_t reduce_kernel_ptr = nullptr;
    switch (op_type) {
        case X86ReduceOpType::kMEAN:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kMEAN>;
            break;
        case X86ReduceOpType::kL1:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kL1>;
            break;
        case X86ReduceOpType::kL2:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kL2>;
            break;
        case X86ReduceOpType::kMIN:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kMIN>;
            break;
        case X86ReduceOpType::kMAX:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kMAX>;
            break;
        case X86ReduceOpType::kSUM:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kSUM>;
            break;
        case X86ReduceOpType::kPROD:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kPROD>;
            break;
        case X86ReduceOpType::kLOGSUM:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kLOGSUM>;
            break;
        case X86ReduceOpType::kLOGSUMEXP:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kLOGSUMEXP>;
            break;
        case X86ReduceOpType::kSUMSQUARE:
            reduce_kernel_ptr = reduce_kernel<X86ReduceOpType::kSUMSQUARE>;
            break;
        default:
            LOGE("Error, unknown binary op_type\n");
            return TNNERR_LAYER_ERR;
    }

    reduce_kernel_ptr(input, output, outer_size, inner_size, reduce_size);

    return TNN_OK;
}

template <int activation_type>
void DepthwiseConvAVX2(float* dst, const float* src, const float* weight, const float* bias, long width, long src_w_step, long fw, long fh,
                   long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep) {
    long dx, fx, fy;
    Float8 bias_v = Float8::loadu(bias);
    Float8 v_zero = Float8(0.f);
    Float8 v_6    = Float8(6.f);
    for (long y = 0; y < height; ++y) {
        auto srcY = src + y * srcHStep;
        auto dstY = dst + y * dstHStep;
        dx        = 0;
        for (; dx + 3 < width; dx += 4) {
            Float8 dst_v[4];
            for (long i = 0; i < 4; i++)
                dst_v[i] = bias_v;
            const auto* src_z    = srcY + src_w_step * dx;
            const auto* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const auto* src_y    = src_z + fy * dilate_y_step;
                const auto* weight_y = weight_z + fy * fw * 8;
                for (fx = 0; fx < fw; ++fx) {
                    Float8 weight_v = Float8::loadu(weight_y + 8 * fx);
                    Float8 src_v0   = Float8::load(src_y + fx * dilate_x_step);
                    Float8 src_v1 = Float8::load(src_y + fx * dilate_x_step + src_w_step);
                    Float8 src_v2 = Float8::load(src_y + fx * dilate_x_step + 2 * src_w_step);
                    Float8 src_v3 = Float8::load(src_y + fx * dilate_x_step + 3 * src_w_step);
                    Float8::mla(dst_v[0], src_v0, weight_v);
                    Float8::mla(dst_v[1], src_v1, weight_v);
                    Float8::mla(dst_v[2], src_v2, weight_v);
                    Float8::mla(dst_v[3], src_v3, weight_v);
                }
            }
            if (activation_type == ActivationType_ReLU || 
                activation_type == ActivationType_ReLU6) {
                dst_v[0] = Float8::max(dst_v[0], v_zero);
                dst_v[1] = Float8::max(dst_v[1], v_zero);
                dst_v[2] = Float8::max(dst_v[2], v_zero);
                dst_v[3] = Float8::max(dst_v[3], v_zero);
            }
            if (activation_type == ActivationType_ReLU6) {
                dst_v[0] = Float8::min(dst_v[0], v_6);
                dst_v[1] = Float8::min(dst_v[1], v_6);
                dst_v[2] = Float8::min(dst_v[2], v_6);
                dst_v[3] = Float8::min(dst_v[3], v_6);
            }
            Float8::save(dstY + (dx + 0) * 8, dst_v[0]);
            Float8::save(dstY + (dx + 1) * 8, dst_v[1]);
            Float8::save(dstY + (dx + 2) * 8, dst_v[2]);
            Float8::save(dstY + (dx + 3) * 8, dst_v[3]);
        }
        for (; dx < width; ++dx) {
            Float8 dst_v = bias_v;
            const auto* src_z    = srcY + src_w_step * dx;
            const auto* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const auto* src_y    = src_z + fy * dilate_y_step;
                const auto* weight_y = weight_z + fy * fw * 8;
                for (fx = 0; fx < fw; ++fx) {
                    Float8 src_v    = Float8::load(src_y + fx * dilate_x_step);
                    Float8 weight_v = Float8::loadu(weight_y + 8 * fx);
                    Float8::mla(dst_v, src_v, weight_v);
                }
            }
            if (activation_type == ActivationType_ReLU || 
                activation_type == ActivationType_ReLU6) {
                dst_v = Float8::max(dst_v, v_zero);
            }
            if (activation_type == ActivationType_ReLU6) {
                dst_v = Float8::min(dst_v, v_6);
            }
            Float8::save(dstY + dx * 8, dst_v);
        }
    }
}

template void DepthwiseConvAVX2<ActivationType_None>(
    float* dst, const float* src, const float* weight, const float* bias, long width, long src_w_step, long fw, long fh,
    long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep);

template void DepthwiseConvAVX2<ActivationType_ReLU>(
    float* dst, const float* src, const float* weight, const float* bias, long width, long src_w_step, long fw, long fh,
    long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep);

template void DepthwiseConvAVX2<ActivationType_ReLU6>(
    float* dst, const float* src, const float* weight, const float* bias, long width, long src_w_step, long fw, long fh,
    long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep);

//ActivationType_SIGMOID_MUL TBD

}