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

#include <algorithm>
#include <cstring>
#include <functional>
#include <type_traits>
#include <iostream>
namespace TNN_NS {

Status X86_BINARY_CALCULATE(const std::vector<void *> &input_ptrs, const std::vector<DimsVector> &input_shapes,
                            Blob *output, std::shared_ptr<X86_BINARY_OP> op) {
    if (input_shapes.size() != 2) {
        return Status(TNNERR_MODEL_ERR, "Error: add layer is invalid.");
    }

    const int batch         = output->GetBlobDesc().dims[0];
    const int channel       = output->GetBlobDesc().dims[1];
    const int height        = output->GetBlobDesc().dims[2];
    const int width         = output->GetBlobDesc().dims[3];
    const int channel_size  = height * width;
    const int count         = batch * channel * channel_size;
    float *output_data      = static_cast<float*>(output->GetHandle().base);
    float *input_data0      = static_cast<float*>(input_ptrs[0]);
    float *input_data1      = static_cast<float*>(input_ptrs[1]);
    auto input_shape0 = input_shapes[0], input_shape1 = input_shapes[1];
    for (int b = 0; b < batch; b++) {
        int output_index_b = b * channel * channel_size;
        int input_index0_b = std::min(b, input_shape0[0] - 1) * input_shape0[1] * input_shape0[2] * input_shape0[3];
        int input_index1_b = std::min(b, input_shape1[0] - 1) * input_shape1[1] * input_shape1[2] * input_shape1[3];

        for (int c = 0; c < channel; c++) {
            int output_index_c = c * channel_size + output_index_b;
            int input_index0_c = std::min(c, input_shape0[1] - 1) * input_shape0[2] * input_shape0[3] + input_index0_b;
            int input_index1_c = std::min(c, input_shape1[1] - 1) * input_shape1[2] * input_shape1[3] + input_index1_b;
            
            for (int h = 0; h < height; h++) {
                int output_index_h = h * width + output_index_c;
                int input_index0_h = std::min(h, input_shape0[2] - 1) * input_shape0[3] + input_index0_c;
                int input_index1_h = std::min(h, input_shape1[2] - 1) * input_shape1[3] + input_index1_c;

                for (int w = 0; w < width; w++) {
                    int output_index = w + output_index_h;
                    int input_index0 = std::min(w, input_shape0[3] - 1) + input_index0_h;
                    int input_index1 = std::min(w, input_shape1[3] - 1) + input_index1_h;
                    output_data[output_index] = (*op)(input_data0[input_index0], input_data1[input_index1]);
                }
            }
        }
    }
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

    // for (int c = 0; c < channel; c++) {
    //     for (int h = 0; h < height_col; h++) {
    //         for (int w = 0; w < width_col; w++) {
    //             for (int kh = 0; kh < kernelh; kh++) {
    //                 for (int kw = 0; kw < kernelw; kw++) {
    //                     int h_pad = h * strideh - padh + kh;
    //                     int w_pad = w * stridew - padw + kw;
    //                     if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
    //                         *dst = src[(c * height + h_pad) * width + w_pad];
    //                         std::cout << (c * height + h_pad) * width + w_pad << std::endl;}
    //                     else
    //                         *dst = 0;
    //                     dst++;
    //                 }
    //             }
    //         }
    //     }
    // }

    return TNN_OK;
}

Status X86_matrixMul(int m, int n, int k, float *A, float *B, float *C,
                     int has_bias, float *bias, int activation_type) {
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
            register __m256 src, scale, bias;
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
            register __m256 src, scale;
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
                    register __m256 src, scale, bias;
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
                    register __m256 src, scale;
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

Status X86_REDUCE_CALCULATE(float *input, float *output, DimsVector input_dim, DimsVector output_dim, std::shared_ptr<X86_REDUCE_OP> op) {

    int channel = input_dim[1];
    int channel_size = DimsVectorUtils::Count(input_dim, 2);
#ifdef __AVX2__
    int tail = channel_size - channel_size % 8;
    register __m256 src_, tmp_;
    for (int b = 0; b < output_dim[0]; b++) {
        for (int index = 0; index < tail; index += 8) {
            tmp_ = _mm256_setzero_ps();
            for (int c = 0; c < channel; c++) {
                src_ = _mm256_loadu_ps(input + (b * channel + c) * channel_size + index);
                tmp_ = (*op)(tmp_, src_);
            }
            _mm256_storeu_ps(output + b * channel_size + index, tmp_);
        }

        for (int index = tail; index < channel_size; index++) {
            float tmp = 0.f;
            for (int c = 0; c < channel; c++) {
                tmp += input[(b * channel + c) * channel_size + index];
            }
            output[b * channel_size + index] = tmp;
        }
        // unsigned a[8] = {0};
        // for (int i = 0; i < channel_size % 8; i++) a[i] = 1;
        // for (int i = 0; i < 8; i++) std::cout << a[i] << std::endl;
        // __m256i mask_ = _mm256_loadu_si256((__m256i*)a);
        
        // for (int index = tail; index < channel_size; index += 8) {
        //     tmp_ = _mm256_setzero_ps();
        //     for (int c = 0; c < channel; c++) {
        //         src_ = _mm256_maskload_ps(input + (b * channel + c) * channel_size + index, mask_);
        //         tmp_ = (*op)(tmp_, src_);
        //     }
        //     _mm256_maskstore_ps(output + b * channel_size + index, mask_, tmp_);
        // }
    }
#else
    for (int b = 0; b < output_dim[0]; b++) {
        for (int index = 0; index < channel_size; index++) {
            op->Init();
            for (int c = 0; c < channel; c++) {
                (*op)(input[(b * channel + c) * channel_size + index]);
            }
            output[b * channel_size + index] = op->GetValue();
        }
    }
#endif
    return TNN_OK;
}

}