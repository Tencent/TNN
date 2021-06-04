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

#ifndef SOURCE_TNN_DEVICE_X86_ACC_X86_COMPUTE_INT8_H_
#define SOURCE_TNN_DEVICE_X86_ACC_X86_COMPUTE_INT8_H_

#include "tnn/core/common.h"
#include "tnn/core/blob.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"

namespace TNN_NS {

void X86AVXGemmInt8Unit4x4(const int8_t* src, const int8_t* weight, int8_t* dst, long src_w_step, long dst_depth, long cdiv8,
                     const float* scale, const int32_t* bias, long relu, const int8_t* add_input,
                     const float* add_scale, const int8_t* relu6_max);

void X86SSEGemmInt8Unit4x4(const int8_t* src, const int8_t* weight, int8_t* dst, long src_w_step, long dst_depth, long cdiv8,
                     const float* scale, const int32_t* bias, long relu, const int8_t* add_input,
                     const float* add_scale, const int8_t* relu6_max);

void X86DepthwiseI8Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, long fw, long fh,
                     long weight_y_step, long dilate_y_step, long dilate_x_step, const float* scale, long dst_depth);

void X86DepthwiseI8General(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
                        long dilate_y_step, long dilate_x_step, long src_w_step, long dst_depth, long fw, long fh,
                        const float* scale_z);

void X86DepthwiseI8K3(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
                   long dilate_y_step, long dialte_x_step, long src_w_step, long dst_depth, long fw, long fh,
                   const float* scale_z);

void X86DepthwiseI8K5(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
                   long dilate_y_step, long dialte_x_step, long src_w_step, long dst_depth, long fw, long fh,
                   const float* scale_z);

void X86ReluInt8(int8_t* dst, const int8_t* src, long len);
void X86Relu6Int8(int8_t* dst, const int8_t* src, const int8_t* relu6_max, long width, long dst_depth);

void X86MaxPoolingINT8(const int8_t* src, long iw, long ih, int8_t* dst, long ow, long oh, long c_r4, long kw, long kh,
                    long stride_w, long stride_h, long pad_w, long pad_h);

void X86AvgPoolingINT8(const int8_t* src, long iw, long ih, int8_t* dst, long ow, long oh, long c_r4, long kw, long kh,
                    long stride_w, long stride_h, long pad_w, long pad_h);

void X86MatrixAddInt8(int8_t* dst, const int8_t* A, const int8_t* B, float* dst_scale, const float* a_scale,
                   float* b_scale, long channel, long hw_size);

void X86GemvInt8(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, const float* scale,
                 long ic_r4, long oc_r4);

void X86ConcatChannelInt8(Blob *output, const std::vector<Blob *> &inputs);
void X86ConcatCommonInt8(Blob *output, const std::vector<Blob *> &inputs, int axis);

template <bool do_scale>
void X86UpsampleNearest2D(int8_t *output_data, const int8_t *input_data,
                          int ih, int iw, int oh, int ow, int c_4, const float *scale);

template <bool do_scale>
void X86UpsampleBilinear2D(int8_t *output_data, const int8_t *input_data,
                           int batch, int ih, int iw, int oh, int ow,
                           int c_4, bool align_corners, const float *scale);

void X86Int8ToFloat(float* dst, const int8_t* src, const float* scale, long batch, long channel, long hw);
void X86FloatToInt8(int8_t* dst, const float* src, const float* scale, long batch, long channel, long hw);

}   // namespace TNN_NS

#endif
