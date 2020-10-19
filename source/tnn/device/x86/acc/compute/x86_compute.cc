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

}