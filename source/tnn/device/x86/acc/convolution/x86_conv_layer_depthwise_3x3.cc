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

#include "tnn/device/x86/acc/convolution/x86_conv_layer_depthwise_3x3.h"
#include "tnn/device/x86/x86_common.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/device/x86/x86_util.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

void conv_dw_3x3s1_avx(const float *din, float *dout, int num, int ch_out, int h_out, int w_out, int ch_in, int h_in,
                       int w_in, int pad, const float *weights, const float *bias, int act_type, X86Context *context) {
  bool right = false;

  float *zero_ptr = (float *)context->GetSharedWorkSpace(MAX(w_in * sizeof(float), 8 * sizeof(float)), 0);
  memset(zero_ptr, 0, MAX(w_in * sizeof(float), 8 * sizeof(float)));
  float *write_ptr = (float *)context->GetSharedWorkSpace(w_out * sizeof(float), 1);

  //! prepare for processing right result
  int rmask_o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float rmaskr[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  int r = w_out % 6;
  int col = w_out / 6;
  if (r > 0) col++;
  if (r > 0) {
    for (int i = 0; i < 8; i++) {
      if (i < r) {
        rmask_o[i] = 0x80000000;
      }
      if (i <= r + (1 - pad)) {
        rmaskr[i] = -1.f;
      }
    }
    right = true;
  } else {
    for (int i = 0; i < 7 + (1 - pad); i++) {
      rmaskr[i] = -1.f;
    }
  }

  __m256i shift_1 = _mm256_set_epi32(7, 7, 6, 5, 4, 3, 2, 1);
  __m256i shift_2 = _mm256_set_epi32(7, 7, 7, 6, 5, 4, 3, 2);
  __m256i shift_3 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  __m256 zero = _mm256_set1_ps(0.f);

  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    for (int c = 0; c < ch_in; c++) {
      float *dout_ptr = dout_batch + c * size_out_channel;
      const float *din_ch_ptr = din_batch + c * size_in_channel;

      __m256 v_bias = _mm256_set1_ps(bias[c]);
      const float *wei_ptr = weights + c * w_stride;

      float *doutr0 = dout_ptr;
      float *doutr1 = doutr0 + w_out;
      float *doutr2 = doutr1 + w_out;
      float *doutr3 = doutr2 + w_out;

      const float *dr0 = din_ch_ptr;
      const float *dr1 = dr0 + w_in;
      const float *dr2 = dr1 + w_in;
      const float *dr3 = dr2 + w_in;
      const float *dr4 = dr3 + w_in;
      const float *dr5 = dr4 + w_in;

      const float *din_ptr0 = dr0;
      const float *din_ptr1 = dr1;
      const float *din_ptr2 = dr2;
      const float *din_ptr3 = dr3;
      const float *din_ptr4 = dr4;
      const float *din_ptr5 = dr5;

      for (int i = 0; i < h_out; i += 4) {
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;
        din_ptr4 = dr4;
        din_ptr5 = dr5;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        doutr2 = doutr1 + w_out;
        doutr3 = doutr2 + w_out;

        //! process top pad
        if (i == 0 && pad == 1) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          din_ptr4 = dr3;
          din_ptr5 = dr4;
          dr0 = dr3;
          dr1 = dr4;
          dr2 = dr5;
        } else {
          dr0 = dr4;
          dr1 = dr5;
          dr2 = dr1 + w_in;
        }
        dr3 = dr2 + w_in;
        dr4 = dr3 + w_in;
        dr5 = dr4 + w_in;

        //! process bottom pad
        if (i + 5 + (1 - pad) > h_in) {
          switch (i + 5 + (1 - pad) - h_in) {
            case 5:
              din_ptr1 = zero_ptr;
            case 4:
              din_ptr2 = zero_ptr;
            case 3:
              din_ptr3 = zero_ptr;
            case 2:
              din_ptr4 = zero_ptr;
            case 1:
              din_ptr5 = zero_ptr;
            default:
              break;
          }
        }

        //! process bottom remain
        if (i + 4 > h_out) {
          switch (i + 4 - h_out) {
            case 3:
              doutr1 = write_ptr;
            case 2:
              doutr2 = write_ptr;
            case 1:
              doutr3 = write_ptr;
            default:
              break;
          }
        }

        for (int j = 0; j < col; j += 1) {
          __m256 i0 = _mm256_loadu_ps(din_ptr0);
          __m256 i1 = _mm256_loadu_ps(din_ptr1);
          __m256 i2 = _mm256_loadu_ps(din_ptr2);
          __m256 i3 = _mm256_loadu_ps(din_ptr3);
          __m256 i4 = _mm256_loadu_ps(din_ptr4);
          __m256 i5 = _mm256_loadu_ps(din_ptr5);

          //! process left pad
          if (j == 0 && pad == 1) {
            din_ptr0 += 5;
            din_ptr1 += 5;
            din_ptr2 += 5;
            din_ptr3 += 5;
            din_ptr4 += 5;
            din_ptr5 += 5;
            i0 = _mm256_blend_ps(zero, i0, 0b01111111);
            i0 = _mm256_permutevar8x32_ps(i0, shift_3);
            i1 = _mm256_blend_ps(zero, i1, 0b01111111);
            i1 = _mm256_permutevar8x32_ps(i1, shift_3);
            i2 = _mm256_blend_ps(zero, i2, 0b01111111);
            i2 = _mm256_permutevar8x32_ps(i2, shift_3);
            i3 = _mm256_blend_ps(zero, i3, 0b01111111);
            i3 = _mm256_permutevar8x32_ps(i3, shift_3);
            i4 = _mm256_blend_ps(zero, i4, 0b01111111);
            i4 = _mm256_permutevar8x32_ps(i4, shift_3);
            i5 = _mm256_blend_ps(zero, i5, 0b01111111);
            i5 = _mm256_permutevar8x32_ps(i5, shift_3);
          } else {
            din_ptr0 += 6;
            din_ptr1 += 6;
            din_ptr2 += 6;
            din_ptr3 += 6;
            din_ptr4 += 6;
            din_ptr5 += 6;
          }

          //! process right remain
          __m256i smask_ = _mm256_set_epi32(0,
                                            0,
                                            0x80000000,
                                            0x80000000,
                                            0x80000000,
                                            0x80000000,
                                            0x80000000,
                                            0x80000000);
          if (j + 1 == col) {
            __m256 rmask_i = _mm256_loadu_ps(rmaskr);
            i0 = _mm256_blendv_ps(zero, i0, rmask_i);
            i1 = _mm256_blendv_ps(zero, i1, rmask_i);
            i2 = _mm256_blendv_ps(zero, i2, rmask_i);
            i3 = _mm256_blendv_ps(zero, i3, rmask_i);
            i4 = _mm256_blendv_ps(zero, i4, rmask_i);
            i5 = _mm256_blendv_ps(zero, i5, rmask_i);
            dout_ptr = dout_ptr + 4 * w_out;
            if (right) {
              smask_ = _mm256_set_epi32(rmask_o[7],
                                        rmask_o[6],
                                        rmask_o[5],
                                        rmask_o[4],
                                        rmask_o[3],
                                        rmask_o[2],
                                        rmask_o[1],
                                        rmask_o[0]);
            }
          }

          __m256 wei_00 = _mm256_set1_ps(*(wei_ptr));
          __m256 wei_01 = _mm256_set1_ps(*(wei_ptr + 1));
          __m256 wei_02 = _mm256_set1_ps(*(wei_ptr + 2));

          // r0 row0
          __m256 r0 = _mm256_fmadd_ps(i0, wei_00, v_bias);
          __m256 tmp = _mm256_permutevar8x32_ps(i0, shift_1);
          r0 = _mm256_fmadd_ps(tmp, wei_01, r0);
          tmp = _mm256_permutevar8x32_ps(i0, shift_2);
          r0 = _mm256_fmadd_ps(tmp, wei_02, r0);

          // r1 row0
          __m256 r1 = _mm256_fmadd_ps(i1, wei_00, v_bias);
          tmp = _mm256_permutevar8x32_ps(i1, shift_1);
          r1 = _mm256_fmadd_ps(tmp, wei_01, r1);
          tmp = _mm256_permutevar8x32_ps(i1, shift_2);
          r1 = _mm256_fmadd_ps(tmp, wei_02, r1);

          // r2 row0
          __m256 r2 = _mm256_fmadd_ps(i2, wei_00, v_bias);
          tmp = _mm256_permutevar8x32_ps(i2, shift_1);
          r2 = _mm256_fmadd_ps(tmp, wei_01, r2);
          tmp = _mm256_permutevar8x32_ps(i2, shift_2);
          r2 = _mm256_fmadd_ps(tmp, wei_02, r2);

          // r3 row0
          __m256 r3 = _mm256_fmadd_ps(i3, wei_00, v_bias);
          tmp = _mm256_permutevar8x32_ps(i3, shift_1);
          r3 = _mm256_fmadd_ps(tmp, wei_01, r3);
          tmp = _mm256_permutevar8x32_ps(i3, shift_2);
          r3 = _mm256_fmadd_ps(tmp, wei_02, r3);

          __m256 wei_10 = _mm256_set1_ps(*(wei_ptr + 3));
          __m256 wei_11 = _mm256_set1_ps(*(wei_ptr + 4));
          __m256 wei_12 = _mm256_set1_ps(*(wei_ptr + 5));

          // r0 row0 + row1
          r0 = _mm256_fmadd_ps(i1, wei_10, r0);
          tmp = _mm256_permutevar8x32_ps(i1, shift_1);
          r0 = _mm256_fmadd_ps(tmp, wei_11, r0);
          tmp = _mm256_permutevar8x32_ps(i1, shift_2);
          r0 = _mm256_fmadd_ps(tmp, wei_12, r0);

          // r1 row0 + row1
          r1 = _mm256_fmadd_ps(i2, wei_10, r1);
          tmp = _mm256_permutevar8x32_ps(i2, shift_1);
          r1 = _mm256_fmadd_ps(tmp, wei_11, r1);
          tmp = _mm256_permutevar8x32_ps(i2, shift_2);
          r1 = _mm256_fmadd_ps(tmp, wei_12, r1);

          // r2 row0 + row1
          r2 = _mm256_fmadd_ps(i3, wei_10, r2);
          tmp = _mm256_permutevar8x32_ps(i3, shift_1);
          r2 = _mm256_fmadd_ps(tmp, wei_11, r2);
          tmp = _mm256_permutevar8x32_ps(i3, shift_2);
          r2 = _mm256_fmadd_ps(tmp, wei_12, r2);

          // r3 row0 + row1
          r3 = _mm256_fmadd_ps(i4, wei_10, r3);
          tmp = _mm256_permutevar8x32_ps(i4, shift_1);
          r3 = _mm256_fmadd_ps(tmp, wei_11, r3);
          tmp = _mm256_permutevar8x32_ps(i4, shift_2);
          r3 = _mm256_fmadd_ps(tmp, wei_12, r3);

          __m256 wei_20 = _mm256_set1_ps(*(wei_ptr + 6));
          __m256 wei_21 = _mm256_set1_ps(*(wei_ptr + 7));
          __m256 wei_22 = _mm256_set1_ps(*(wei_ptr + 8));

          // r0 row0 + row1 + row2
          r0 = _mm256_fmadd_ps(i2, wei_20, r0);
          tmp = _mm256_permutevar8x32_ps(i2, shift_1);
          r0 = _mm256_fmadd_ps(tmp, wei_21, r0);
          tmp = _mm256_permutevar8x32_ps(i2, shift_2);
          r0 = _mm256_fmadd_ps(tmp, wei_22, r0);

          // r1 row0 + row1 + row2
          r1 = _mm256_fmadd_ps(i3, wei_20, r1);
          tmp = _mm256_permutevar8x32_ps(i3, shift_1);
          r1 = _mm256_fmadd_ps(tmp, wei_21, r1);
          tmp = _mm256_permutevar8x32_ps(i3, shift_2);
          r1 = _mm256_fmadd_ps(tmp, wei_22, r1);

          // r2 row0 + row1 + row2
          r2 = _mm256_fmadd_ps(i4, wei_20, r2);
          tmp = _mm256_permutevar8x32_ps(i4, shift_1);
          r2 = _mm256_fmadd_ps(tmp, wei_21, r2);
          tmp = _mm256_permutevar8x32_ps(i4, shift_2);
          r2 = _mm256_fmadd_ps(tmp, wei_22, r2);

          // r3 row0 + row1 + row2
          r3 = _mm256_fmadd_ps(i5, wei_20, r3);
          tmp = _mm256_permutevar8x32_ps(i5, shift_1);
          r3 = _mm256_fmadd_ps(tmp, wei_21, r3);
          tmp = _mm256_permutevar8x32_ps(i5, shift_2);
          r3 = _mm256_fmadd_ps(tmp, wei_22, r3);

          if (act_type == ActivationType_ReLU) {
            r0 = _mm256_max_ps(r0, zero);
            r1 = _mm256_max_ps(r1, zero);
            r2 = _mm256_max_ps(r2, zero);
            r3 = _mm256_max_ps(r3, zero);
          } else if (act_type == ActivationType_ReLU6) {
            __m256 six = _mm256_set1_ps(6.f);
            r0 = _mm256_min_ps(_mm256_max_ps(r0, zero), six);
            r1 = _mm256_min_ps(_mm256_max_ps(r1, zero), six);
            r2 = _mm256_min_ps(_mm256_max_ps(r2, zero), six);
            r3 = _mm256_min_ps(_mm256_max_ps(r3, zero), six);
          }

          _mm256_maskstore_ps(doutr0, smask_, r0);
          _mm256_maskstore_ps(doutr1, smask_, r1);
          _mm256_maskstore_ps(doutr2, smask_, r2);
          _mm256_maskstore_ps(doutr3, smask_, r3);

          doutr0 = doutr0 + 6;
          doutr1 = doutr1 + 6;
          doutr2 = doutr2 + 6;
          doutr3 = doutr3 + 6;
        }
      }
    }
  }
}
void conv_dw_3x3s1_sse(const float *din, float *dout, int num, int ch_out, int h_out, int w_out, int ch_in, int h_in,
                       int w_in, int pad, const float *weights, const float *bias, int act_type, X86Context *context) {
  bool right = false;  // for right result

  float *zero_ptr = (float *)context->GetSharedWorkSpace(MAX(w_in * sizeof(float), 8 * sizeof(float)), 0);
  memset(zero_ptr, 0, MAX(w_in * sizeof(float), 8 * sizeof(float)));
  float *write_ptr = (float *)context->GetSharedWorkSpace(w_out * sizeof(float), 1);

  //! prepare for processing right result
  float rmasko[4] = {1.f, 1.f, 1.f, 1.f};
  float rmaskr[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  int r = w_out % 4;
  int col = w_out / 4;
  if (r > 0) col++;
  if (r > 0) {
    for (int i = 0; i < 4; i++) {
      if (i < r) {
        rmasko[i] = -1.f;
      }
    }
    right = true;
  }
  if (r > 0) {
    for (int i = 0; i < 8; i++) {
      if (i <= r + 1 - pad) {
        rmaskr[i] = -1.f;
      }
    }
  } else {
    for (int i = 0; i < 5 + (1 - pad); i++) {
      rmaskr[i] = -1.f;
    }
  }

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  __m128 zero = _mm_set1_ps(0.f);

  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    for (int c = 0; c < ch_in; c++) {
      float *dout_ptr = dout_batch + c * size_out_channel;
      const float *din_ch_ptr = din_batch + c * size_in_channel;

      __m128 v_bias = _mm_set1_ps(bias[c]);
      const float *wei_ptr = weights + c * w_stride;

      float *doutr0 = dout_ptr;
      float *doutr1 = doutr0 + w_out;

      const float *dr0 = din_ch_ptr;
      const float *dr1 = dr0 + w_in;
      const float *dr2 = dr1 + w_in;
      const float *dr3 = dr2 + w_in;

      const float *din_ptr0 = dr0;
      const float *din_ptr1 = dr1;
      const float *din_ptr2 = dr2;
      const float *din_ptr3 = dr3;

      for (int i = 0; i < h_out; i += 2) {
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;

        //! process top pad
        if (i == 0 && pad == 1) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          dr0 = dr1;
        } else {
          dr0 = dr2;
        }
        dr1 = dr0 + w_in;
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;

        //! process bottom pad
        if (i + 3 + (1 - pad) > h_in) {
          switch (i + 3 + (1 - pad) - h_in) {
            case 3:
              din_ptr1 = zero_ptr;
            case 2:
              din_ptr2 = zero_ptr;
            case 1:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          switch (i + 2 - h_out) {
            case 1:
              doutr1 = write_ptr;
            default:
              break;
          }
        }

        for (int j = 0; j < col; j += 1) {
          __m128 i0_0 = _mm_loadu_ps(din_ptr0);
          __m128 i0_1 = _mm_loadu_ps(din_ptr0 + 4);
          __m128 i1_0 = _mm_loadu_ps(din_ptr1);
          __m128 i1_1 = _mm_loadu_ps(din_ptr1 + 4);
          __m128 i2_0 = _mm_loadu_ps(din_ptr2);
          __m128 i2_1 = _mm_loadu_ps(din_ptr2 + 4);
          __m128 i3_0 = _mm_loadu_ps(din_ptr3);
          __m128 i3_1 = _mm_loadu_ps(din_ptr3 + 4);

          //! process left pad
          if (j == 0 && pad == 1) {
            __m128 tmp0 = _mm_blend_ps(zero, i0_0, 0b0111);
            tmp0 = _mm_shuffle_ps(tmp0, tmp0, 0b10010011);
            i0_1 = _mm_blend_ps(i0_0, i0_1, 0b0111);
            i0_1 = _mm_shuffle_ps(i0_1, i0_1, 0b10010011);
            i0_0 = tmp0;

            tmp0 = _mm_blend_ps(zero, i1_0, 0b0111);
            tmp0 = _mm_shuffle_ps(tmp0, tmp0, 0b10010011);
            i1_1 = _mm_blend_ps(i1_0, i1_1, 0b0111);
            i1_1 = _mm_shuffle_ps(i1_1, i1_1, 0b10010011);
            i1_0 = tmp0;

            tmp0 = _mm_blend_ps(zero, i2_0, 0b0111);
            tmp0 = _mm_shuffle_ps(tmp0, tmp0, 0b10010011);
            i2_1 = _mm_blend_ps(i2_0, i2_1, 0b0111);
            i2_1 = _mm_shuffle_ps(i2_1, i2_1, 0b10010011);
            i2_0 = tmp0;

            tmp0 = _mm_blend_ps(zero, i3_0, 0b0111);
            tmp0 = _mm_shuffle_ps(tmp0, tmp0, 0b10010011);
            i3_1 = _mm_blend_ps(i3_0, i3_1, 0b0111);
            i3_1 = _mm_shuffle_ps(i3_1, i3_1, 0b10010011);
            i3_0 = tmp0;

            din_ptr0 += 3;
            din_ptr1 += 3;
            din_ptr2 += 3;
            din_ptr3 += 3;
          } else {
            din_ptr0 += 4;
            din_ptr1 += 4;
            din_ptr2 += 4;
            din_ptr3 += 4;
          }

          //! process right remain
          if (j + 1 == col) {
            __m128 rmask_i = _mm_loadu_ps(rmaskr);
            i0_0 = _mm_blendv_ps(zero, i0_0, rmask_i);
            i1_0 = _mm_blendv_ps(zero, i1_0, rmask_i);
            i2_0 = _mm_blendv_ps(zero, i2_0, rmask_i);
            i3_0 = _mm_blendv_ps(zero, i3_0, rmask_i);

            rmask_i = _mm_loadu_ps(rmaskr + 4);
            i0_1 = _mm_blendv_ps(zero, i0_1, rmask_i);
            i1_1 = _mm_blendv_ps(zero, i1_1, rmask_i);
            i2_1 = _mm_blendv_ps(zero, i2_1, rmask_i);
            i3_1 = _mm_blendv_ps(zero, i3_1, rmask_i);

            dout_ptr = dout_ptr + 2 * w_out;
          }

          __m128 wei_00 = _mm_load1_ps(wei_ptr);
          __m128 wei_01 = _mm_load1_ps(wei_ptr + 1);
          __m128 wei_02 = _mm_load1_ps(wei_ptr + 2);

          // r0 row0
          __m128 r0 = _mm_mul_ps(i0_0, wei_00);
          r0 = _mm_add_ps(r0, v_bias);
          __m128 tmp = _mm_blend_ps(i0_0, i0_1, 0b0001);
          tmp = _mm_shuffle_ps(tmp, tmp, 0b00111001);
          tmp = _mm_mul_ps(tmp, wei_01);
          r0 = _mm_add_ps(tmp, r0);
          tmp = _mm_blend_ps(i0_0, i0_1, 0b0011);
          tmp = _mm_shuffle_ps(tmp, tmp, 0b01001110);
          tmp = _mm_mul_ps(tmp, wei_02);
          r0 = _mm_add_ps(tmp, r0);

          // r1 row0
          __m128 r1 = _mm_mul_ps(i1_0, wei_00);
          r1 = _mm_add_ps(r1, v_bias);
          tmp = _mm_blend_ps(i1_0, i1_1, 0b0001);
          tmp = _mm_shuffle_ps(tmp, tmp, 0b00111001);
          tmp = _mm_mul_ps(tmp, wei_01);
          r1 = _mm_add_ps(tmp, r1);
          tmp = _mm_blend_ps(i1_0, i1_1, 0b0011);
          tmp = _mm_shuffle_ps(tmp, tmp, 0b01001110);
          tmp = _mm_mul_ps(tmp, wei_02);
          r1 = _mm_add_ps(tmp, r1);

          __m128 wei_10 = _mm_load1_ps(wei_ptr + 3);
          __m128 wei_11 = _mm_load1_ps(wei_ptr + 4);
          __m128 wei_12 = _mm_load1_ps(wei_ptr + 5);

          // r0 row0 + row1
          tmp = _mm_mul_ps(i1_0, wei_10);
          r0 = _mm_add_ps(r0, tmp);
          tmp = _mm_blend_ps(i1_0, i1_1, 0b0001);
          tmp = _mm_shuffle_ps(tmp, tmp, 0b00111001);
          tmp = _mm_mul_ps(tmp, wei_11);
          r0 = _mm_add_ps(tmp, r0);
          tmp = _mm_blend_ps(i1_0, i1_1, 0b0011);
          tmp = _mm_shuffle_ps(tmp, tmp, 0b01001110);
          tmp = _mm_mul_ps(tmp, wei_12);
          r0 = _mm_add_ps(tmp, r0);

          // r1 row0 + row1
          tmp = _mm_mul_ps(i2_0, wei_10);
          r1 = _mm_add_ps(r1, tmp);
          tmp = _mm_blend_ps(i2_0, i2_1, 0b0001);
          tmp = _mm_shuffle_ps(tmp, tmp, 0b00111001);
          tmp = _mm_mul_ps(tmp, wei_11);
          r1 = _mm_add_ps(tmp, r1);
          tmp = _mm_blend_ps(i2_0, i2_1, 0b0011);
          tmp = _mm_shuffle_ps(tmp, tmp, 0b01001110);
          tmp = _mm_mul_ps(tmp, wei_12);
          r1 = _mm_add_ps(tmp, r1);

          __m128 wei_20 = _mm_load1_ps(wei_ptr + 6);
          __m128 wei_21 = _mm_load1_ps(wei_ptr + 7);
          __m128 wei_22 = _mm_load1_ps(wei_ptr + 8);

          // r0 row0 + row1 + row2
          tmp = _mm_mul_ps(i2_0, wei_20);
          r0 = _mm_add_ps(r0, tmp);
          tmp = _mm_blend_ps(i2_0, i2_1, 0b0001);
          tmp = _mm_shuffle_ps(tmp, tmp, 0b00111001);
          tmp = _mm_mul_ps(tmp, wei_21);
          r0 = _mm_add_ps(tmp, r0);
          tmp = _mm_blend_ps(i2_0, i2_1, 0b0011);
          tmp = _mm_shuffle_ps(tmp, tmp, 0b01001110);
          tmp = _mm_mul_ps(tmp, wei_22);
          r0 = _mm_add_ps(tmp, r0);

          // r1 row0 + row1 + row2
          tmp = _mm_mul_ps(i3_0, wei_20);
          r1 = _mm_add_ps(r1, tmp);
          tmp = _mm_blend_ps(i3_0, i3_1, 0b0001);
          tmp = _mm_shuffle_ps(tmp, tmp, 0b00111001);
          tmp = _mm_mul_ps(tmp, wei_21);
          r1 = _mm_add_ps(tmp, r1);
          tmp = _mm_blend_ps(i3_0, i3_1, 0b0011);
          tmp = _mm_shuffle_ps(tmp, tmp, 0b01001110);
          tmp = _mm_mul_ps(tmp, wei_22);
          r1 = _mm_add_ps(tmp, r1);

          if (act_type == ActivationType_ReLU) {
              r0 = _mm_max_ps(r0, zero);
              r1 = _mm_max_ps(r1, zero);
            } else if (act_type == ActivationType_ReLU6) {
              __m128 six = _mm_set1_ps(6.f);
              r0 = _mm_min_ps(_mm_max_ps(r0, zero), six);
              r1 = _mm_min_ps(_mm_max_ps(r1, zero), six);
          }

          //! process bottom pad
          if (j + 1 == col && right) {
            __m128 out0 = _mm_loadu_ps(doutr0);
            __m128 out1 = _mm_loadu_ps(doutr1);
            __m128 rmask_ro = _mm_loadu_ps(rmasko);
            r0 = _mm_blendv_ps(out0, r0, rmask_ro);
            r1 = _mm_blendv_ps(out1, r1, rmask_ro);
          }

          _mm_storeu_ps(doutr0, r0);
          _mm_storeu_ps(doutr1, r1);

          doutr0 += 4;
          doutr1 += 4;
        }
      }
    }
  }
}
void conv_dw_3x3s2_avx(const float *din, float *dout, int num, int ch_out, int h_out, int w_out, int ch_in, int h_in,
                       int w_in, int pad, const float *weights, const float *bias, int act_type, X86Context *context) {
  bool right = false;  // for right result
      
  float *zero_ptr = (float *)context->GetSharedWorkSpace(MAX(w_in * sizeof(float), 8 * sizeof(float)), 0);
  memset(zero_ptr, 0, MAX(w_in * sizeof(float), 8 * sizeof(float)));
  float *write_ptr = (float *)context->GetSharedWorkSpace(w_out * sizeof(float), 1);

  //! prepare for processing right result
  int rmask_o[4] = {0};
  float rmaskr[8] = {-1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f};
  int ro = w_out % 3;
  int col = w_out / 3;
  if (ro > 0) col++;
  if (ro > 0) {
    for (int i = 0; i < 4; i++) {
      if (i < ro) {
        rmask_o[i] = 0x80000000;
      }
    }
    right = true;
  }
  int ri = (w_in - (1 - pad)) % 6;
  // [pad == 0 && w_out == 3 && win == 8] ===>>> [ri == 1 && ro == 0]
  // add condition ro > 0 for avoiding wrong rmaskr when pad == 0
  if (ri > 0 && (ro > 0 || pad == 1)) {
    for (int i = 0; i < 8; i++) {
      if (i <= ri) {
        rmaskr[i] = -1.f;
      } else {
        rmaskr[i] = 1.f;
      }
    }
  }

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  __m128 zero = _mm_set1_ps(0.f);
  __m256 zero_256 = _mm256_set1_ps(0.f);

  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    for (int c = 0; c < ch_in; c++) {
      float *dout_ptr = dout_batch + c * size_out_channel;
      const float *din_ch_ptr = din_batch + c * size_in_channel;

      __m256 v_bias = _mm256_set1_ps(bias[c]);
      const float *wei_ptr = weights + c * w_stride;

      const float *dr0 = din_ch_ptr;
      const float *dr1 = dr0 + w_in;
      const float *dr2 = dr1 + w_in;
      const float *dr3 = dr2 + w_in;
      const float *dr4 = dr3 + w_in;

      const float *din_ptr0 = dr0;
      const float *din_ptr1 = dr1;
      const float *din_ptr2 = dr2;
      const float *din_ptr3 = dr3;
      const float *din_ptr4 = dr4;

      float *doutr0 = dout_ptr;
      float *doutr1 = doutr0 + w_out;

      // for shift input
      __m256i shift_0 = _mm256_set_epi32(7, 7, 6, 5, 4, 3, 2, 1);
      __m256i shift_1 = _mm256_set_epi32(7, 7, 7, 6, 5, 4, 3, 2);
      __m256i shift_3 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);

      for (int i = 0; i + (1 - pad) < h_in; i += 4) {
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;
        din_ptr4 = dr4;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;

        //! process top pad
        if (i == 0 && pad == 1) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          din_ptr4 = dr3;
          dr0 = dr3;
          dr1 = dr0 + w_in;
        } else {
          dr0 = dr4;
          dr1 = dr0 + w_in;
        }
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;
        dr4 = dr3 + w_in;

        //! process bottom pad
        if (i + 4 + (1 - pad) > h_in) {
          switch (i + 4 + (1 - pad) - h_in) {
            case 4:
              din_ptr1 = zero_ptr;
            case 3:
              din_ptr2 = zero_ptr;
            case 2:
              din_ptr3 = zero_ptr;
            case 1:
              din_ptr4 = zero_ptr;
            default:
              break;
          }
        }

        //! process bottom remain
        if (i / 2 + 2 > h_out) {
          switch (i / 2 + 2 - h_out) {
            case 2:
              doutr0 = write_ptr;
            case 1:
              doutr1 = write_ptr;
            default:
              break;
          }
        }

        for (int j = 0; j < col; j += 1) {
          __m256 i0 = _mm256_loadu_ps(din_ptr0);
          __m256 i2 = _mm256_loadu_ps(din_ptr2);
          __m256 i1 = _mm256_loadu_ps(din_ptr1);
          __m256 i3 = _mm256_loadu_ps(din_ptr3);
          __m256 i4 = _mm256_loadu_ps(din_ptr4);

          //! process left pad
          if (j == 0 && pad == 1) {
            din_ptr0 += 5;
            din_ptr1 += 5;
            din_ptr2 += 5;
            din_ptr3 += 5;
            din_ptr4 += 5;
            i0 = _mm256_blend_ps(zero_256, i0, 0b01111111);
            i0 = _mm256_permutevar8x32_ps(i0, shift_3);
            i1 = _mm256_blend_ps(zero_256, i1, 0b01111111);
            i1 = _mm256_permutevar8x32_ps(i1, shift_3);
            i2 = _mm256_blend_ps(zero_256, i2, 0b01111111);
            i2 = _mm256_permutevar8x32_ps(i2, shift_3);
            i3 = _mm256_blend_ps(zero_256, i3, 0b01111111);
            i3 = _mm256_permutevar8x32_ps(i3, shift_3);
            i4 = _mm256_blend_ps(zero_256, i4, 0b01111111);
            i4 = _mm256_permutevar8x32_ps(i4, shift_3);
          } else {
            din_ptr0 += 6;
            din_ptr1 += 6;
            din_ptr2 += 6;
            din_ptr3 += 6;
            din_ptr4 += 6;
          }

          //! process right remain
          __m128i mask = _mm_setr_epi32(0x80000000, 0x80000000, 0x80000000, 0);
          if (j + 1 == col) {
            __m256 rmask_ri = _mm256_loadu_ps(rmaskr);
            i0 = _mm256_blendv_ps(zero_256, i0, rmask_ri);
            i1 = _mm256_blendv_ps(zero_256, i1, rmask_ri);
            i2 = _mm256_blendv_ps(zero_256, i2, rmask_ri);
            i3 = _mm256_blendv_ps(zero_256, i3, rmask_ri);
            i4 = _mm256_blendv_ps(zero_256, i4, rmask_ri);
            dout_ptr = dout_ptr + 2 * w_out;
            if (right) {
              mask = _mm_setr_epi32(
                  rmask_o[0], rmask_o[1], rmask_o[2], rmask_o[3]);
            }
          }

          __m256 wei_00 = _mm256_set1_ps(*(wei_ptr));
          __m256 wei_01 = _mm256_set1_ps(*(wei_ptr + 1));
          __m256 wei_02 = _mm256_set1_ps(*(wei_ptr + 2));

          // r0 row0
          __m256 res0 = _mm256_fmadd_ps(i0, wei_00, v_bias);
          __m256 tmp = _mm256_permutevar8x32_ps(i0, shift_0);
          res0 = _mm256_fmadd_ps(tmp, wei_01, res0);
          tmp = _mm256_permutevar8x32_ps(i0, shift_1);
          res0 = _mm256_fmadd_ps(tmp, wei_02, res0);

          // r1 row0
          __m256 res1 = _mm256_fmadd_ps(i2, wei_00, v_bias);
          tmp = _mm256_permutevar8x32_ps(i2, shift_0);
          res1 = _mm256_fmadd_ps(tmp, wei_01, res1);
          tmp = _mm256_permutevar8x32_ps(i2, shift_1);
          res1 = _mm256_fmadd_ps(tmp, wei_02, res1);

          __m256 wei_10 = _mm256_set1_ps(*(wei_ptr + 3));
          __m256 wei_11 = _mm256_set1_ps(*(wei_ptr + 4));
          __m256 wei_12 = _mm256_set1_ps(*(wei_ptr + 5));

          // r0 row0 + row1
          res0 = _mm256_fmadd_ps(i1, wei_10, res0);
          tmp = _mm256_permutevar8x32_ps(i1, shift_0);
          res0 = _mm256_fmadd_ps(tmp, wei_11, res0);
          tmp = _mm256_permutevar8x32_ps(i1, shift_1);
          res0 = _mm256_fmadd_ps(tmp, wei_12, res0);

          // r1 row0 + row1
          res1 = _mm256_fmadd_ps(i3, wei_10, res1);
          tmp = _mm256_permutevar8x32_ps(i3, shift_0);
          res1 = _mm256_fmadd_ps(tmp, wei_11, res1);
          tmp = _mm256_permutevar8x32_ps(i3, shift_1);
          res1 = _mm256_fmadd_ps(tmp, wei_12, res1);

          __m256 wei_20 = _mm256_set1_ps(*(wei_ptr + 6));
          __m256 wei_21 = _mm256_set1_ps(*(wei_ptr + 7));
          __m256 wei_22 = _mm256_set1_ps(*(wei_ptr + 8));

          // r0 row0 + row1 + row2
          res0 = _mm256_fmadd_ps(i2, wei_20, res0);
          tmp = _mm256_permutevar8x32_ps(i2, shift_0);
          res0 = _mm256_fmadd_ps(tmp, wei_21, res0);
          tmp = _mm256_permutevar8x32_ps(i2, shift_1);
          res0 = _mm256_fmadd_ps(tmp, wei_22, res0);

          // r1 row0 + row1 + row2
          res1 = _mm256_fmadd_ps(i4, wei_20, res1);
          tmp = _mm256_permutevar8x32_ps(i4, shift_0);
          res1 = _mm256_fmadd_ps(tmp, wei_21, res1);
          tmp = _mm256_permutevar8x32_ps(i4, shift_1);
          res1 = _mm256_fmadd_ps(tmp, wei_22, res1);

          __m256i shift_2 = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
          __m256 r0 = _mm256_permutevar8x32_ps(res0, shift_2);
          __m128 r0_128 = _mm256_extractf128_ps(r0, 0);

          __m256 r1 = _mm256_permutevar8x32_ps(res1, shift_2);
          __m128 r1_128 = _mm256_extractf128_ps(r1, 0);

          if (act_type == ActivationType_ReLU) {
            r0_128 = _mm_max_ps(r0_128, zero);
            r1_128 = _mm_max_ps(r1_128, zero);
          } else if (act_type == ActivationType_ReLU6) {
            __m128 six = _mm_set1_ps(6.f);
            r0_128 = _mm_min_ps(_mm_max_ps(r0_128, zero), six);
            r1_128 = _mm_min_ps(_mm_max_ps(r1_128, zero), six);
          }

          _mm_maskstore_ps(doutr0, mask, r0_128);
          _mm_maskstore_ps(doutr1, mask, r1_128);

          doutr0 = doutr0 + 3;
          doutr1 = doutr1 + 3;
        }
      }
    }
  }
}
void conv_dw_3x3s2_sse(const float *din, float *dout, int num, int ch_out, int h_out, int w_out, int ch_in, int h_in,
                       int w_in, int pad, const float *weights, const float *bias, int act_type, X86Context *context) {
  bool right = false;  // for right result

  float *zero_ptr = (float *)context->GetSharedWorkSpace(MAX(w_in * sizeof(float), 12 * sizeof(float)), 0);
  memset(zero_ptr, 0, MAX(w_in * sizeof(float), 12 * sizeof(float)));
  float *write_ptr = (float *)context->GetSharedWorkSpace(w_out * sizeof(float), 1);

  //! prepare for processing right result
  float rmasko[4] = {1.f, 1.f, 1.f, 1.f};
  float rmaskr[12] = {
      -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f};
  int ro = w_out % 4;
  int col = w_out / 4;
  if (ro > 0) col++;
  if (ro > 0) {
    for (int i = 0; i < 4; i++) {
      if (i < ro) {
        rmasko[i] = -1.f;
      }
    }
    right = true;
  }
  int ri = (w_in - (1 - pad)) % 8;
  if (ri > 0 && (ro > 0 || pad == 1)) {
    for (int i = 0; i < 12; i++) {
      if (i <= ri) {
        rmaskr[i] = -1.f;
      } else {
        rmaskr[i] = 1.f;
      }
    }
  }

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  __m128 zero = _mm_set1_ps(0.f);

  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    for (int c = 0; c < ch_in; c++) {
      float *dout_ptr = dout_batch + c * size_out_channel;
      const float *din_ch_ptr = din_batch + c * size_in_channel;

      __m128 v_bias = _mm_set1_ps(bias[c]);
      const float *wei_ptr = weights + c * w_stride;

      const float *dr0 = din_ch_ptr;
      const float *dr1 = dr0 + w_in;
      const float *dr2 = dr1 + w_in;

      const float *din_ptr0 = dr0;
      const float *din_ptr1 = dr1;
      const float *din_ptr2 = dr2;

      float *doutr0 = dout_ptr;
      float *doutr0_ptr = doutr0;

      for (int i = 0; i + (1 - pad) < h_in; i += 2) {
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;

        doutr0_ptr = doutr0;

        //! process top pad
        if (i == 0 && pad == 1) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          dr0 = dr1;
          dr1 = dr2;
          dr2 = dr1 + w_in;
        } else {
          dr0 = dr2;
          dr1 = dr0 + w_in;
          dr2 = dr1 + w_in;
        }

        //! process bottom pad
        if (i + 2 + (1 - pad) > h_in) {
          switch (i + 2 + (1 - pad) - h_in) {
            case 2:
              din_ptr1 = zero_ptr;
            case 1:
              din_ptr2 = zero_ptr;
            default:
              break;
          }
        }

        if (i / 2 + 1 > h_out) {
          doutr0_ptr = write_ptr;
        }

        for (int j = 0; j < col; j++) {
          __m128 i0_0 = _mm_loadu_ps(din_ptr0);
          __m128 i0_1 = _mm_loadu_ps(din_ptr0 + 4);
          __m128 i0_2 = _mm_loadu_ps(din_ptr0 + 8);

          __m128 i1_0 = _mm_loadu_ps(din_ptr1);
          __m128 i1_1 = _mm_loadu_ps(din_ptr1 + 4);
          __m128 i1_2 = _mm_loadu_ps(din_ptr1 + 8);

          __m128 i2_0 = _mm_loadu_ps(din_ptr2);
          __m128 i2_1 = _mm_loadu_ps(din_ptr2 + 4);
          __m128 i2_2 = _mm_loadu_ps(din_ptr2 + 8);

          //! process left pad
          if (j == 0 && pad == 1) {
            __m128 tmp0 = _mm_blend_ps(zero, i0_0, 0b0111);
            tmp0 = _mm_shuffle_ps(tmp0, tmp0, 0b10010011);
            __m128 tmp1 = _mm_blend_ps(i0_0, i0_1, 0b0111);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0b10010011);
            i0_2 = _mm_blend_ps(i0_1, i0_2, 0b0111);
            i0_2 = _mm_shuffle_ps(i0_2, i0_2, 0b10010011);
            i0_0 = tmp0;
            i0_1 = tmp1;

            tmp0 = _mm_blend_ps(zero, i1_0, 0b0111);
            tmp0 = _mm_shuffle_ps(tmp0, tmp0, 0b10010011);
            tmp1 = _mm_blend_ps(i1_0, i1_1, 0b0111);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0b10010011);
            i1_2 = _mm_blend_ps(i1_1, i1_2, 0b0111);
            i1_2 = _mm_shuffle_ps(i1_2, i1_2, 0b10010011);
            i1_0 = tmp0;
            i1_1 = tmp1;

            tmp0 = _mm_blend_ps(zero, i2_0, 0b0111);
            tmp0 = _mm_shuffle_ps(tmp0, tmp0, 0b10010011);
            tmp1 = _mm_blend_ps(i2_0, i2_1, 0b0111);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0b10010011);
            i2_2 = _mm_blend_ps(i2_1, i2_2, 0b0111);
            i2_2 = _mm_shuffle_ps(i2_2, i2_2, 0b10010011);
            i2_0 = tmp0;
            i2_1 = tmp1;

            din_ptr0 += 7;
            din_ptr1 += 7;
            din_ptr2 += 7;
          } else {
            din_ptr0 += 8;
            din_ptr1 += 8;
            din_ptr2 += 8;
          }

          //! process right remain
          if (j + 1 == col) {
            doutr0 = doutr0 + w_out;
            __m128 rmask = _mm_loadu_ps(rmaskr);
            i0_0 = _mm_blendv_ps(zero, i0_0, rmask);
            i1_0 = _mm_blendv_ps(zero, i1_0, rmask);
            i2_0 = _mm_blendv_ps(zero, i2_0, rmask);

            rmask = _mm_loadu_ps(rmaskr + 4);
            i0_1 = _mm_blendv_ps(zero, i0_1, rmask);
            i1_1 = _mm_blendv_ps(zero, i1_1, rmask);
            i2_1 = _mm_blendv_ps(zero, i2_1, rmask);

            rmask = _mm_loadu_ps(rmaskr + 8);
            i0_2 = _mm_blendv_ps(zero, i0_2, rmask);
            i1_2 = _mm_blendv_ps(zero, i1_2, rmask);
            i2_2 = _mm_blendv_ps(zero, i2_2, rmask);
          }
          //！ shift input
          // 0,1,2,3  4,5,6,7  8,9,10,11 => 0,1,2,3  2,3,4,5  3,4,5,6
          __m128 tmp = _mm_shuffle_ps(i0_0, i0_1, 0b10001000);
          i0_1 = _mm_shuffle_ps(i0_0, i0_1, 0b11011101);
          i0_0 = tmp;
          i0_2 = _mm_blend_ps(i0_2, i0_0, 0b1110);
          i0_2 = _mm_shuffle_ps(i0_2, i0_2, 0b00111001);

          tmp = _mm_shuffle_ps(i1_0, i1_1, 0b10001000);
          i1_1 = _mm_shuffle_ps(i1_0, i1_1, 0b11011101);
          i1_0 = tmp;
          i1_2 = _mm_blend_ps(i1_2, i1_0, 0b1110);
          i1_2 = _mm_shuffle_ps(i1_2, i1_2, 0b00111001);

          tmp = _mm_shuffle_ps(i2_0, i2_1, 0b10001000);
          i2_1 = _mm_shuffle_ps(i2_0, i2_1, 0b11011101);
          i2_0 = tmp;
          i2_2 = _mm_blend_ps(i2_2, i2_0, 0b1110);
          i2_2 = _mm_shuffle_ps(i2_2, i2_2, 0b00111001);

          __m128 wei_00 = _mm_load1_ps(wei_ptr);
          __m128 wei_01 = _mm_load1_ps(wei_ptr + 1);
          __m128 wei_02 = _mm_load1_ps(wei_ptr + 2);

          // r0 row0
          __m128 r0 = _mm_mul_ps(i0_0, wei_00);
          r0 = _mm_add_ps(r0, v_bias);
          tmp = _mm_mul_ps(i0_1, wei_01);
          r0 = _mm_add_ps(r0, tmp);
          tmp = _mm_mul_ps(i0_2, wei_02);
          r0 = _mm_add_ps(r0, tmp);

          __m128 wei_10 = _mm_load1_ps(wei_ptr + 3);
          __m128 wei_11 = _mm_load1_ps(wei_ptr + 4);
          __m128 wei_12 = _mm_load1_ps(wei_ptr + 5);

          // r0 row0 + row1
          tmp = _mm_mul_ps(i1_0, wei_10);
          r0 = _mm_add_ps(r0, tmp);
          tmp = _mm_mul_ps(i1_1, wei_11);
          r0 = _mm_add_ps(r0, tmp);
          tmp = _mm_mul_ps(i1_2, wei_12);
          r0 = _mm_add_ps(r0, tmp);

          __m128 wei_20 = _mm_load1_ps(wei_ptr + 6);
          __m128 wei_21 = _mm_load1_ps(wei_ptr + 7);
          __m128 wei_22 = _mm_load1_ps(wei_ptr + 8);

          // r0 row0 + row1 + row2
          tmp = _mm_mul_ps(i2_0, wei_20);
          r0 = _mm_add_ps(r0, tmp);
          tmp = _mm_mul_ps(i2_1, wei_21);
          r0 = _mm_add_ps(r0, tmp);
          tmp = _mm_mul_ps(i2_2, wei_22);
          r0 = _mm_add_ps(r0, tmp);

          if (act_type == ActivationType_ReLU) {
              r0 = _mm_max_ps(r0, zero);
            } else if (act_type == ActivationType_ReLU6) {
              __m128 six = _mm_set1_ps(6.f);
              r0 = _mm_min_ps(_mm_max_ps(r0, zero), six);
          }

          //! process bottom pad
          if (j + 1 == col && right) {
            __m128 out0 = _mm_loadu_ps(doutr0_ptr);
            __m128 rmask_ro = _mm_loadu_ps(rmasko);
            r0 = _mm_blendv_ps(out0, r0, rmask_ro);
          }

          _mm_storeu_ps(doutr0_ptr, r0);

          doutr0_ptr += 4;
        }
      }
    }
  }
}

bool X86ConvLayerDepthwise3x3::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                          const std::vector<Blob *> &outputs) {
                                            return false;
    if (!param) {
        return false;
    }

    const int group          = param->group;
    const int input_channel  = inputs[0]->GetBlobDesc().dims[1];
    const int output_channel = outputs[0]->GetBlobDesc().dims[1];

    bool group_cond    = (group == input_channel) && (group == output_channel);
    bool kernel_cond   = (param->kernels[0] == 3) && (param->kernels[1] == 3);
    bool dilation_cond = (param->dialations[0] == 1) && (param->dialations[1] == 1);
    bool pad_cond      = (param->pads[0] == param->pads[2]) && (param->pads[0] / 2 == 0);
    bool stride_cond   = (param->strides[0] == param->strides[1]) && (param->strides[0] == 1 || param->strides[0] == 2);

    return group_cond && kernel_cond && dilation_cond && pad_cond && stride_cond;
}

X86ConvLayerDepthwise3x3::~X86ConvLayerDepthwise3x3() {}

Status X86ConvLayerDepthwise3x3::allocateBufferWeight(const std::vector<Blob *> &inputs,
                                                      const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    if (!buffer_weight_.GetBytesSize()) {
        int kw = param->kernels[0];
        int kh = param->kernels[1];

        const int group  = param->group;
        const float *src = conv_res->filter_handle.force_to<float *>();

        int weight_count   = group * kh * kw;
        int data_byte_size = DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());

        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
            RawBuffer temp_buffer(weight_count * data_byte_size);
            float *dst = temp_buffer.force_to<float *>();

            // no weights trans, just copy
            memcpy(dst, src, kw * kh * group * sizeof(float));

            temp_buffer.SetDataType(DATA_TYPE_FLOAT);
            buffer_weight_ = temp_buffer;
        } else {
            LOGE("Error: DataType %d not support\n", conv_res->filter_handle.GetDataType());
            return Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        }
    }
    return TNN_OK;
}

Status X86ConvLayerDepthwise3x3::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    int c_pack = 8;
    if (arch_ == sse42) {
        c_pack = 4;
    }

    const int batch = dims_output[0];

    const float *src_origin = reinterpret_cast<const float *>(input->GetHandle().base);
    float *dst_origin       = reinterpret_cast<float *>(output->GetHandle().base);

    auto dw_func = conv_dw_3x3s1_avx;
    if (param->strides[0] == 2) {
        dw_func = conv_dw_3x3s2_avx;
    }
    if (arch_ == sse42) {
        if (param->strides[0] == 1) {
            dw_func = conv_dw_3x3s1_sse;
        } else {
            dw_func = conv_dw_3x3s2_sse;
        }
    }

    dw_func(src_origin, dst_origin, batch, dims_output[1], dims_output[2], dims_output[3], dims_input[1], dims_input[2],
            dims_input[3], param->pads[0], buffer_weight_.force_to<float *>(), buffer_bias_.force_to<float *>(),
            param->activation_type, context_);

    return TNN_OK;
}

}  // namespace TNN_NS
