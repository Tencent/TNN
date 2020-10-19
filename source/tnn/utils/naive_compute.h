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

#ifndef TNN_UTILS_NAIVE_COMPUTE_H_
#define TNN_UTILS_NAIVE_COMPUTE_H_

#include <float.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/interpreter/layer_param.h"

namespace TNN_NS {

int8_t float2int8(float val);
uint8_t float2uint8(float val);

template <typename T, typename Tacc>
void NaivePooling(T *input_ptr, T *output_ptr, DimsVector dims_input, DimsVector dims_output, 
                int stride_y, int stride_x, int kernel_y, int kernel_x, int pad_y, int pad_x, int pool_type);

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void NaiveConv(void *input_ptr, void *output_ptr, void *weight_ptr, void *bias, DimsVector dims_input,
            DimsVector dims_output, int stride_y, int stride_x, int kernel_size_y, int kernel_size_x, int pad_y,
            int pad_x, int group, int dilation, int activation_type, float *scale, int scale_len);

// float fc
template <typename T>
void NaiveFC(T *input_ptr, T *output_ptr, T *weight_data, float *bias, DimsVector dims_input, DimsVector dims_output);

// int8 fc: reload by scale and scale_len
void NaiveFC(void *input_ptr, void *output_ptr, void *weight_data, float *scale, int scale_len, void *bias,
            DimsVector dims_input, DimsVector dims_output);

/**
 * @brief Permute the input blob by changing the memory order of the data.
 **/
template <typename T>
void NaivePermute(const int count, T *bottom_data, const std::vector<int> &permute_order,
                const std::vector<int> &old_steps, const std::vector<int> &new_steps, const int num_axes, T *top_data);

void NaiveReorg(float *bottom_data, int w, int h, int c, int batch, int stride, int reverse, int mode, float *top_data);

void NaivePriorbox(PriorBoxLayerParam *param, int output_h, int output_w, float *output_data, int layer_height,
                   int layer_width, int img_height, int img_width, float step_h, float step_w);

void priorbox_set_value(const int N, const float alpha, float *Y);

void NaiveDetectionOutput(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                          DetectionOutputLayerParam *param);

void NaiveBGROrBGRAToGray(const uint8_t* src, uint8_t* dst, int h, int w, int channel);

void NaiveYUVToBGR(const unsigned char* yuv, unsigned char* bgr, int h, int w, bool is_nv12);

void NaiveYUVToBGRA(const unsigned char* yuv, unsigned char* bgra, int h, int w, bool is_nv12);

void NaiveYUVToBGROrBGRALoop(const unsigned char *yptr0, const unsigned char *yptr1, const unsigned char *vuptr,
                             unsigned char* rgb0, unsigned char* rgb1, int remain, bool is_nv12, int channel);

void NaiveYUVToBGROrBGRA(const unsigned char* yuv, unsigned char* bgr, const int channel, const int h, const int w, bool is_nv12);

void NaiveDequant(const int8_t *input_ptr, const float *scale_ptr, int scale_len, float *output, DimsVector dims);

void NaiveQuant(const float *input_ptr, const float *scale_ptr, int scale_len, int8_t *output, DimsVector dims);

}  // namespace TNN_NS

#endif  // TNN_UTILS_NAIVE_COMPUTE_H_
