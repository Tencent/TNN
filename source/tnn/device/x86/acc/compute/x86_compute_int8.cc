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

#include "tnn/device/x86/acc/compute/x86_compute_int8.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/device/x86/x86_common.h"

namespace TNN_NS {

void X86GemmInt8Unit4x4(const int8_t* src, const int8_t* weight, int8_t* dst, long src_w_step, long dst_depth, long cdiv8,
                     const float* scale, const int32_t* bias, long relu, const int8_t* add_input,
                     const float* add_scale, const int8_t* relu6_max) {
    // set rounding mode for _mm_cvtps_pi8
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);

    __m128 zero_f32 = _mm_set1_ps(0.f);
    __m128 add_05   = _mm_set1_ps(0.5f);
    __m128 sub_05   = _mm_set1_ps(-0.5f);
    __m128i zero_i8 = _mm_setzero_si128();
    __m128 relu6_max_vec;

    if (relu == 2) {
        float tmp4[4];
        tmp4[0] = (float)relu6_max[0];
        tmp4[1] = (float)relu6_max[1];
        tmp4[2] = (float)relu6_max[2];
        tmp4[3] = (float)relu6_max[3];
        relu6_max_vec = _mm_loadu_ps(tmp4);
    }

    for (long w = 0; w < 4; ++w) {
        const auto src_x   = src + w * src_w_step;
        auto dst_x         = dst + w * dst_depth;
        auto add_input_x   = add_input ? add_input + w * dst_depth : nullptr;
        int32_t dstTemp[4] = {0, 0, 0, 0};
        long sz            = 0;

        __m128i dst_vec_0 = _mm_setzero_si128();
        __m128i dst_vec_1 = _mm_setzero_si128();
        __m128i dst_vec_2 = _mm_setzero_si128();
        __m128i dst_vec_3 = _mm_setzero_si128();

        for (; sz < cdiv8 / 2; ++sz) {
            const auto weight_sz = weight + (4 * 16) * sz;
            const auto src_z     = src_x + sz * 16;

            __m128i w_vec0  = _mm_loadu_si128((__m128i*)(weight_sz));
            __m128i w_vec1  = _mm_loadu_si128((__m128i*)(weight_sz + 16));
            __m128i w_vec2  = _mm_loadu_si128((__m128i*)(weight_sz + 32));
            __m128i w_vec3  = _mm_loadu_si128((__m128i*)(weight_sz + 48));
            __m128i src_vec = _mm_loadu_si128((__m128i*)(src_z));

            __m128i w_16_00 = _mm_cvtepi8_epi16(_mm_unpacklo_epi64(w_vec0, w_vec0));
            __m128i w_16_01 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(w_vec0, w_vec0));
            __m128i w_16_10 = _mm_cvtepi8_epi16(_mm_unpacklo_epi64(w_vec1, w_vec1));
            __m128i w_16_11 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(w_vec1, w_vec1));
            __m128i w_16_20 = _mm_cvtepi8_epi16(_mm_unpacklo_epi64(w_vec2, w_vec2));
            __m128i w_16_21 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(w_vec2, w_vec2));
            __m128i w_16_30 = _mm_cvtepi8_epi16(_mm_unpacklo_epi64(w_vec3, w_vec3));
            __m128i w_16_31 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(w_vec3, w_vec3));

            __m128i src_16_0 = _mm_cvtepi8_epi16(_mm_unpacklo_epi64(src_vec, src_vec));
            __m128i src_16_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(src_vec, src_vec));

            __m128i d_32_00 = _mm_madd_epi16(w_16_00, src_16_0);
            __m128i d_32_01 = _mm_madd_epi16(w_16_01, src_16_1);
            __m128i d_32_10 = _mm_madd_epi16(w_16_10, src_16_0);
            __m128i d_32_11 = _mm_madd_epi16(w_16_11, src_16_1);
            __m128i d_32_20 = _mm_madd_epi16(w_16_20, src_16_0);
            __m128i d_32_21 = _mm_madd_epi16(w_16_21, src_16_1);
            __m128i d_32_30 = _mm_madd_epi16(w_16_30, src_16_0);
            __m128i d_32_31 = _mm_madd_epi16(w_16_31, src_16_1);

            dst_vec_0 = _mm_add_epi32(dst_vec_0, d_32_00);
            dst_vec_1 = _mm_add_epi32(dst_vec_1, d_32_10);
            dst_vec_2 = _mm_add_epi32(dst_vec_2, d_32_20);
            dst_vec_3 = _mm_add_epi32(dst_vec_3, d_32_30);
            dst_vec_0 = _mm_add_epi32(dst_vec_0, d_32_01);
            dst_vec_1 = _mm_add_epi32(dst_vec_1, d_32_11);
            dst_vec_2 = _mm_add_epi32(dst_vec_2, d_32_21);
            dst_vec_3 = _mm_add_epi32(dst_vec_3, d_32_31);
        }

        for (; sz < cdiv8 / 2 + cdiv8 % 2; ++sz) {
            const auto weight_sz = weight + (4 * 16) * sz;
            const auto src_z     = src_x + sz * 16;

            __m128i w_vec0  = _mm_loadu_si64(weight_sz);
            __m128i w_vec1  = _mm_loadu_si64(weight_sz + 16);
            __m128i w_vec2  = _mm_loadu_si64(weight_sz + 32);
            __m128i w_vec3  = _mm_loadu_si64(weight_sz + 48);
            __m128i src_vec = _mm_loadu_si64(src_z);

            __m128i w_16_00 = _mm_cvtepi8_epi16(w_vec0);
            __m128i w_16_10 = _mm_cvtepi8_epi16(w_vec1);
            __m128i w_16_20 = _mm_cvtepi8_epi16(w_vec2);
            __m128i w_16_30 = _mm_cvtepi8_epi16(w_vec3);

            __m128i src_16_0 = _mm_cvtepi8_epi16(src_vec);

            __m128i d_32_00 = _mm_madd_epi16(w_16_00, src_16_0);
            __m128i d_32_10 = _mm_madd_epi16(w_16_10, src_16_0);
            __m128i d_32_20 = _mm_madd_epi16(w_16_20, src_16_0);
            __m128i d_32_30 = _mm_madd_epi16(w_16_30, src_16_0);

            dst_vec_0 = _mm_add_epi32(dst_vec_0, d_32_00);
            dst_vec_1 = _mm_add_epi32(dst_vec_1, d_32_10);
            dst_vec_2 = _mm_add_epi32(dst_vec_2, d_32_20);
            dst_vec_3 = _mm_add_epi32(dst_vec_3, d_32_30);
        }

        dst_vec_0 = _mm_hadd_epi32(dst_vec_0, dst_vec_1);
        dst_vec_1 = _mm_hadd_epi32(dst_vec_2, dst_vec_3);
        dst_vec_0 = _mm_hadd_epi32(dst_vec_0, dst_vec_1);

        __m128i bias_vec = _mm_loadu_si128((__m128i*)bias);
        __m128 scale_vec = _mm_loadu_ps(scale);
        __m128 dst_4x32   = _mm_cvtepi32_ps(_mm_add_epi32(dst_vec_0, bias_vec));
        dst_4x32          = _mm_mul_ps(dst_4x32, scale_vec);

        if (relu == -1) {
            dst_4x32 = _mm_max_ps(dst_4x32, zero_f32);
        }
        if (add_input_x) {
            int add_input_4x8 = *((int*)(add_input_x));
            __m128 add_scale_vec = _mm_loadu_ps(add_scale);
            __m128 add_input_vec = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(add_input_4x8)));
            dst_4x32 = _mm_add_ps(dst_4x32, _mm_mul_ps(add_input_vec, add_scale_vec));
        }
        if (relu == 1) {
            dst_4x32 = _mm_max_ps(dst_4x32, zero_f32);
        }
        // Conv-Add-Relu6
        else if (relu == 2) {
            dst_4x32 = _mm_max_ps(dst_4x32, zero_f32);
            dst_4x32 = _mm_min_ps(dst_4x32, relu6_max_vec);
        }

        // rounding to zero(val + (val >= 0.f ? 0.5f : -0.5f)) = rounding to nearest ties away from zero
        __m128 cmp_zero = _mm_cmpge_ps(dst_4x32, zero_f32);
        __m128 adjust_vec = _mm_blendv_ps(sub_05, add_05, cmp_zero);
        dst_4x32 = _mm_add_ps(dst_4x32, adjust_vec);
        __m128i dst_int8 = _mm_movpi64_epi64(_mm_cvtps_pi8(dst_4x32));
        int dst_4x8 = _mm_extract_epi32(dst_int8, 0);
        *((int*)(dst_x)) = dst_4x8;
    }
}

static void DepthwiseI8K3Kernel(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z,
                                long src_y_step, long src_w_step, long dst_depth, const float* scale_z,
                                long dx, long dc) {
    // set rounding mode for _mm_cvtps_pi8
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);

    __m128i zero_i8 = _mm_setzero_si128();
    __m128 zero_f32 = _mm_set1_ps(0.f);
    __m128 add_05   = _mm_set1_ps(0.5f);
    __m128 sub_05   = _mm_set1_ps(-0.5f);

    auto dst_x       = dst + dx * dst_depth + dc;
    const auto src_z = src + dx * src_w_step + dc;
    __m128i bias_vec0 = _mm_loadu_si128((__m128i*)(bias_z + dc));
    __m128i bias_vec1 = _mm_loadu_si128((__m128i*)(bias_z + dc + 4));

    for (long fy = 0; fy < 3; ++fy) {
        const auto src_y    = src_z + fy * src_y_step;
        const auto weight_y = weight + fy * 3 * dst_depth + dc;

        __m128i src_vec_0 = _mm_loadu_si64(src_y);
        __m128i src_vec_1 = _mm_loadu_si64(src_y + dst_depth);
        __m128i src_vec_2 = _mm_loadu_si64(src_y + 2 * dst_depth);
        __m128i w_vec_0   = _mm_loadu_si64(weight_y);
        __m128i w_vec_1   = _mm_loadu_si64(weight_y + dst_depth);
        __m128i w_vec_2   = _mm_loadu_si64(weight_y + 2 * dst_depth);

        __m128i src_16_0  = _mm_cvtepi8_epi16(src_vec_0);
        __m128i src_16_1  = _mm_cvtepi8_epi16(src_vec_1);
        __m128i src_16_2  = _mm_cvtepi8_epi16(src_vec_2);
        __m128i w_16_0    = _mm_cvtepi8_epi16(w_vec_0);
        __m128i w_16_1    = _mm_cvtepi8_epi16(w_vec_1);
        __m128i w_16_2    = _mm_cvtepi8_epi16(w_vec_2);

        __m128i w_16_00   = _mm_unpacklo_epi16(w_16_0, zero_i8);
        __m128i w_16_01   = _mm_unpackhi_epi16(w_16_0, zero_i8);
        __m128i w_16_10   = _mm_unpacklo_epi16(w_16_1, zero_i8);
        __m128i w_16_11   = _mm_unpackhi_epi16(w_16_1, zero_i8);
        __m128i w_16_20   = _mm_unpacklo_epi16(w_16_2, zero_i8);
        __m128i w_16_21   = _mm_unpackhi_epi16(w_16_2, zero_i8);

        __m128i src_16_00 = _mm_unpacklo_epi16(src_16_0, zero_i8);
        __m128i src_16_01 = _mm_unpackhi_epi16(src_16_0, zero_i8);
        __m128i src_16_10 = _mm_unpacklo_epi16(src_16_1, zero_i8);
        __m128i src_16_11 = _mm_unpackhi_epi16(src_16_1, zero_i8);
        __m128i src_16_20 = _mm_unpacklo_epi16(src_16_2, zero_i8);
        __m128i src_16_21 = _mm_unpackhi_epi16(src_16_2, zero_i8);

        bias_vec0 = _mm_add_epi32(bias_vec0, _mm_madd_epi16(w_16_00, src_16_00));
        bias_vec1 = _mm_add_epi32(bias_vec1, _mm_madd_epi16(w_16_01, src_16_01));
        bias_vec0 = _mm_add_epi32(bias_vec0, _mm_madd_epi16(w_16_10, src_16_10));
        bias_vec1 = _mm_add_epi32(bias_vec1, _mm_madd_epi16(w_16_11, src_16_11));
        bias_vec0 = _mm_add_epi32(bias_vec0, _mm_madd_epi16(w_16_20, src_16_20));
        bias_vec1 = _mm_add_epi32(bias_vec1, _mm_madd_epi16(w_16_21, src_16_21));
    }
    __m128 scale_vec0 = _mm_loadu_ps(scale_z + dc);
    __m128 scale_vec1 = _mm_loadu_ps(scale_z + dc + 4);

    __m128 dst_4x32_0 = _mm_cvtepi32_ps(bias_vec0);
    __m128 dst_4x32_1 = _mm_cvtepi32_ps(bias_vec1);
    dst_4x32_0        = _mm_mul_ps(dst_4x32_0, scale_vec0);
    dst_4x32_1        = _mm_mul_ps(dst_4x32_1, scale_vec1);

    // rounding to zero(val + (val >= 0.f ? 0.5f : -0.5f)) = rounding to nearest ties away from zero
    __m128 cmp_zero_0 = _mm_cmpge_ps(dst_4x32_0, zero_f32);
    __m128 cmp_zero_1 = _mm_cmpge_ps(dst_4x32_1, zero_f32);
    __m128 adjust_vec_0 = _mm_blendv_ps(sub_05, add_05, cmp_zero_0);
    __m128 adjust_vec_1 = _mm_blendv_ps(sub_05, add_05, cmp_zero_1);
    dst_4x32_0 = _mm_add_ps(dst_4x32_0, adjust_vec_0);
    dst_4x32_1 = _mm_add_ps(dst_4x32_1, adjust_vec_1);
    __m128i dst_8x8 = _mm_set_epi64(_mm_cvtps_pi8(dst_4x32_1), _mm_cvtps_pi8(dst_4x32_0));
    dst_8x8 = _mm_shuffle_epi32(dst_8x8, 8);
    _mm_storeu_si64(dst_x, dst_8x8);
}

void DepthwiseI8K5Kernel(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z,
                         long src_y_step, long src_w_step, long dst_depth, const float* scale_z,
                         long dx, long dc) {
    // set rounding mode for _mm_cvtps_pi8
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);

    __m128i zero_i8 = _mm_setzero_si128();
    __m128 zero_f32 = _mm_set1_ps(0.f);
    __m128 add_05   = _mm_set1_ps(0.5f);
    __m128 sub_05   = _mm_set1_ps(-0.5f);

    auto dst_x       = dst + dx * dst_depth + dc;
    const auto src_z = src + dx * src_w_step + dc;
    __m128i bias_vec0 = _mm_loadu_si128((__m128i*)(bias_z + dc));
    __m128i bias_vec1 = _mm_loadu_si128((__m128i*)(bias_z + dc + 4));

    for (long fy = 0; fy < 5; ++fy) {
        const auto src_y    = src_z + fy * src_y_step;
        const auto weight_y = weight + fy * 5 * dst_depth + dc;

        __m128i src_vec_0 = _mm_loadu_si64(src_y);
        __m128i src_vec_1 = _mm_loadu_si64(src_y + dst_depth);
        __m128i src_vec_2 = _mm_loadu_si64(src_y + 2 * dst_depth);
        __m128i src_vec_3 = _mm_loadu_si64(src_y + 3 * dst_depth);
        __m128i src_vec_4 = _mm_loadu_si64(src_y + 4 * dst_depth);
        __m128i w_vec_0   = _mm_loadu_si64(weight_y);
        __m128i w_vec_1   = _mm_loadu_si64(weight_y + dst_depth);
        __m128i w_vec_2   = _mm_loadu_si64(weight_y + 2 * dst_depth);
        __m128i w_vec_3   = _mm_loadu_si64(weight_y + 3 * dst_depth);
        __m128i w_vec_4   = _mm_loadu_si64(weight_y + 4 * dst_depth);

        __m128i src_16_0  = _mm_cvtepi8_epi16(src_vec_0);
        __m128i src_16_1  = _mm_cvtepi8_epi16(src_vec_1);
        __m128i src_16_2  = _mm_cvtepi8_epi16(src_vec_2);
        __m128i src_16_3  = _mm_cvtepi8_epi16(src_vec_3);
        __m128i src_16_4  = _mm_cvtepi8_epi16(src_vec_4);

        __m128i w_16_0    = _mm_cvtepi8_epi16(w_vec_0);
        __m128i w_16_1    = _mm_cvtepi8_epi16(w_vec_1);
        __m128i w_16_2    = _mm_cvtepi8_epi16(w_vec_2);
        __m128i w_16_3    = _mm_cvtepi8_epi16(w_vec_3);
        __m128i w_16_4    = _mm_cvtepi8_epi16(w_vec_4);

        __m128i w_16_00   = _mm_unpacklo_epi16(w_16_0, zero_i8);
        __m128i w_16_01   = _mm_unpackhi_epi16(w_16_0, zero_i8);
        __m128i w_16_10   = _mm_unpacklo_epi16(w_16_1, zero_i8);
        __m128i w_16_11   = _mm_unpackhi_epi16(w_16_1, zero_i8);
        __m128i w_16_20   = _mm_unpacklo_epi16(w_16_2, zero_i8);
        __m128i w_16_21   = _mm_unpackhi_epi16(w_16_2, zero_i8);
        __m128i w_16_30   = _mm_unpacklo_epi16(w_16_3, zero_i8);
        __m128i w_16_31   = _mm_unpackhi_epi16(w_16_3, zero_i8);
        __m128i w_16_40   = _mm_unpacklo_epi16(w_16_4, zero_i8);
        __m128i w_16_41   = _mm_unpackhi_epi16(w_16_4, zero_i8);

        __m128i src_16_00 = _mm_unpacklo_epi16(src_16_0, zero_i8);
        __m128i src_16_01 = _mm_unpackhi_epi16(src_16_0, zero_i8);
        __m128i src_16_10 = _mm_unpacklo_epi16(src_16_1, zero_i8);
        __m128i src_16_11 = _mm_unpackhi_epi16(src_16_1, zero_i8);
        __m128i src_16_20 = _mm_unpacklo_epi16(src_16_2, zero_i8);
        __m128i src_16_21 = _mm_unpackhi_epi16(src_16_2, zero_i8);
        __m128i src_16_30 = _mm_unpacklo_epi16(src_16_3, zero_i8);
        __m128i src_16_31 = _mm_unpackhi_epi16(src_16_3, zero_i8);
        __m128i src_16_40 = _mm_unpacklo_epi16(src_16_4, zero_i8);
        __m128i src_16_41 = _mm_unpackhi_epi16(src_16_4, zero_i8);

        bias_vec0 = _mm_add_epi32(bias_vec0, _mm_madd_epi16(w_16_00, src_16_00));
        bias_vec1 = _mm_add_epi32(bias_vec1, _mm_madd_epi16(w_16_01, src_16_01));
        bias_vec0 = _mm_add_epi32(bias_vec0, _mm_madd_epi16(w_16_10, src_16_10));
        bias_vec1 = _mm_add_epi32(bias_vec1, _mm_madd_epi16(w_16_11, src_16_11));
        bias_vec0 = _mm_add_epi32(bias_vec0, _mm_madd_epi16(w_16_20, src_16_20));
        bias_vec1 = _mm_add_epi32(bias_vec1, _mm_madd_epi16(w_16_21, src_16_21));
        bias_vec0 = _mm_add_epi32(bias_vec0, _mm_madd_epi16(w_16_30, src_16_30));
        bias_vec1 = _mm_add_epi32(bias_vec1, _mm_madd_epi16(w_16_31, src_16_31));
        bias_vec0 = _mm_add_epi32(bias_vec0, _mm_madd_epi16(w_16_40, src_16_40));
        bias_vec1 = _mm_add_epi32(bias_vec1, _mm_madd_epi16(w_16_41, src_16_41));
    }
    __m128 scale_vec0 = _mm_loadu_ps(scale_z + dc);
    __m128 scale_vec1 = _mm_loadu_ps(scale_z + dc + 4);

    __m128 dst_4x32_0 = _mm_cvtepi32_ps(bias_vec0);
    __m128 dst_4x32_1 = _mm_cvtepi32_ps(bias_vec1);
    dst_4x32_0        = _mm_mul_ps(dst_4x32_0, scale_vec0);
    dst_4x32_1        = _mm_mul_ps(dst_4x32_1, scale_vec1);

    // rounding to zero(val + (val >= 0.f ? 0.5f : -0.5f)) = rounding to nearest ties away from zero
    __m128 cmp_zero_0 = _mm_cmpge_ps(dst_4x32_0, zero_f32);
    __m128 cmp_zero_1 = _mm_cmpge_ps(dst_4x32_1, zero_f32);
    __m128 adjust_vec_0 = _mm_blendv_ps(sub_05, add_05, cmp_zero_0);
    __m128 adjust_vec_1 = _mm_blendv_ps(sub_05, add_05, cmp_zero_1);
    dst_4x32_0 = _mm_add_ps(dst_4x32_0, adjust_vec_0);
    dst_4x32_1 = _mm_add_ps(dst_4x32_1, adjust_vec_1);
    __m128i dst_8x8 = _mm_set_epi64(_mm_cvtps_pi8(dst_4x32_1), _mm_cvtps_pi8(dst_4x32_0));
    dst_8x8 = _mm_shuffle_epi32(dst_8x8, 8);
    _mm_storeu_si64(dst_x, dst_8x8);
}

void X86DepthwiseI8K3(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
                   long dilate_y_step, long dialte_x_step, long src_w_step, long dst_depth, long fw, long fh,
                   const float* scale_z) {
    // general k3 process, calc left dx
    for (long dx = 0; dx < width; dx++) {
        long dc = 0;
        for (; dc < dst_depth - 7; dc += 8) {
            DepthwiseI8K3Kernel(dst, src, weight, bias_z, dilate_y_step, src_w_step, dst_depth, scale_z,
                                dx, dc);
        }

        if (dc < dst_depth) {
            dc = dst_depth - 8;
            DepthwiseI8K3Kernel(dst, src, weight, bias_z, dilate_y_step, src_w_step, dst_depth, scale_z,
                                dx, dc);
        }
    }
}

void X86DepthwiseI8K5(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
                   long dilate_y_step, long dialte_x_step, long src_w_step, long dst_depth, long fw, long fh,
                   const float* scale_z) {
    // general k3 process, calc left dx
    for (long dx = 0; dx < width; dx++) {
        long dc = 0;
        for (; dc < dst_depth - 7; dc += 8) {
            DepthwiseI8K5Kernel(dst, src, weight, bias_z, dilate_y_step, src_w_step, dst_depth, scale_z,
                                dx, dc);
        }

        if (dc < dst_depth) {
            dc = dst_depth - 8;
            DepthwiseI8K5Kernel(dst, src, weight, bias_z, dilate_y_step, src_w_step, dst_depth, scale_z,
                                dx, dc);
        }
    }
}

/*
convdw int8 kernel, used in corner process
*/
void X86DepthwiseI8Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, long fw, long fh,
                     long weight_y_step, long dilate_y_step, long dilate_x_step, const float* scale, long dst_depth) {
    // set rounding mode for _mm_cvtps_pi8
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);

    __m128i zero_i8 = _mm_setzero_si128();
    __m128 zero_f32 = _mm_set1_ps(0.f);
    __m128 add_05   = _mm_set1_ps(0.5f);
    __m128 sub_05   = _mm_set1_ps(-0.5f);
    long dc = 0;
    for (; dc < dst_depth - 4; dc += 8) {
        __m128i bias_vec0 = _mm_loadu_si128((__m128i*)(bias + dc));
        __m128i bias_vec1 = _mm_loadu_si128((__m128i*)(bias + dc + 4));
        for (long fy = 0; fy < fh; ++fy) {
            const auto src_y    = src + fy * dilate_y_step + dc;
            const auto weight_y = weight + fy * weight_y_step + dc;
            for (long fx = 0; fx < fw; ++fx) {
                const auto src_x    = src_y + fx * dilate_x_step;
                const auto weight_x = weight_y + dst_depth * fx;

                __m128i w_vec    = _mm_loadu_si64(weight_x);
                __m128i src_vec  = _mm_loadu_si64(src_x);
                __m128i w_16     = _mm_cvtepi8_epi16(w_vec);
                __m128i src_16   = _mm_cvtepi8_epi16(src_vec);

                // w_vec   = [w0, 0, w1, 0, w2, 0, w3, 0]
                // src_vec = [s0, 0, s1, 0, s2, 0, s3, 0]
                __m128i w_16_0   = _mm_unpacklo_epi16(w_16, zero_i8);
                __m128i w_16_1   = _mm_unpackhi_epi16(w_16, zero_i8);
                __m128i src_16_0 = _mm_unpacklo_epi16(src_16, zero_i8);
                __m128i src_16_1 = _mm_unpackhi_epi16(src_16, zero_i8);

                __m128i dst_0    = _mm_madd_epi16(w_16_0, src_16_0);
                __m128i dst_1    = _mm_madd_epi16(w_16_1, src_16_1);

                bias_vec0        = _mm_add_epi32(bias_vec0, dst_0);
                bias_vec1        = _mm_add_epi32(bias_vec1, dst_1);
            }
        }
        __m128 scale_vec0 = _mm_loadu_ps(scale + dc);
        __m128 scale_vec1 = _mm_loadu_ps(scale + dc + 4);

        __m128 dst_4x32_0 = _mm_cvtepi32_ps(bias_vec0);
        __m128 dst_4x32_1 = _mm_cvtepi32_ps(bias_vec1);
        dst_4x32_0        = _mm_mul_ps(dst_4x32_0, scale_vec0);
        dst_4x32_1        = _mm_mul_ps(dst_4x32_1, scale_vec1);

        // rounding to zero(val + (val >= 0.f ? 0.5f : -0.5f)) = rounding to nearest ties away from zero
        __m128 cmp_zero_0 = _mm_cmpge_ps(dst_4x32_0, zero_f32);
        __m128 cmp_zero_1 = _mm_cmpge_ps(dst_4x32_1, zero_f32);
        __m128 adjust_vec_0 = _mm_blendv_ps(sub_05, add_05, cmp_zero_0);
        __m128 adjust_vec_1 = _mm_blendv_ps(sub_05, add_05, cmp_zero_1);
        dst_4x32_0 = _mm_add_ps(dst_4x32_0, adjust_vec_0);
        dst_4x32_1 = _mm_add_ps(dst_4x32_1, adjust_vec_1);
        __m128i dst_8x8 = _mm_set_epi64(_mm_cvtps_pi8(dst_4x32_1), _mm_cvtps_pi8(dst_4x32_0));
        dst_8x8 = _mm_shuffle_epi32(dst_8x8, 8);
        _mm_storeu_si64(dst + dc, dst_8x8);
    }
    for (; dc < dst_depth; dc += 4) {
        long dst_temp[4] = {0, 0, 0, 0};
        for (long fy = 0; fy < fh; ++fy) {
            const auto src_y    = src + fy * dilate_y_step + dc;
            const auto weight_y = weight + fy * weight_y_step + dc;
            for (long fx = 0; fx < fw; ++fx) {
                const auto weight_x = weight_y + fx * dst_depth;
                const auto src_x    = src_y + fx * dilate_x_step;
                for (long j = 0; j < 4; ++j) {
                    dst_temp[j] += (int32_t)src_x[j] * (int32_t)weight_x[j];
                }
            }
        }
        for (long i = 0; i < 4; ++i) {
            dst[dc + i] = float2int8(static_cast<float>(dst_temp[i] + bias[dc + i]) * scale[dc + i]);
        }
    }
}

/*
general convdw int8 func
*/
void X86DepthwiseI8General(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
                        long dilate_y_step, long dilate_x_step, long src_w_step, long dst_depth, long fw, long fh,
                        const float* scale_z) {
    // set rounding mode for _mm_cvtps_pi8
    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);

    __m128i zero_i8 = _mm_setzero_si128();
    __m128 zero_f32 = _mm_set1_ps(0.f);
    __m128 add_05   = _mm_set1_ps(0.5f);
    __m128 sub_05   = _mm_set1_ps(-0.5f);

    long dx, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        long dc = 0;
        for (; dc < dst_depth - 4; dc += 8) {
            auto dst_x       = dst + dx * dst_depth + dc;
            const auto src_z = src + dx * src_w_step + dc;
            __m128i bias_vec0 = _mm_loadu_si128((__m128i*)(bias_z + dc));
            __m128i bias_vec1 = _mm_loadu_si128((__m128i*)(bias_z + dc + 4));

            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilate_y_step;
                const auto weight_y = weight + fy * fw * dst_depth + dc;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilate_x_step;
                    const auto weight_x = weight_y + dst_depth * fx;

                    __m128i w_vec    = _mm_loadu_si64(weight_x);
                    __m128i src_vec  = _mm_loadu_si64(src_x);
                    __m128i w_16     = _mm_cvtepi8_epi16(w_vec);
                    __m128i src_16   = _mm_cvtepi8_epi16(src_vec);
                    __m128i w_16_0   = _mm_unpacklo_epi16(w_16, zero_i8);
                    __m128i w_16_1   = _mm_unpackhi_epi16(w_16, zero_i8);
                    __m128i src_16_0 = _mm_unpacklo_epi16(src_16, zero_i8);
                    __m128i src_16_1 = _mm_unpackhi_epi16(src_16, zero_i8);
                    __m128i dst_0    = _mm_madd_epi16(w_16_0, src_16_0);
                    __m128i dst_1    = _mm_madd_epi16(w_16_1, src_16_1);
                    bias_vec0        = _mm_add_epi32(bias_vec0, dst_0);
                    bias_vec1        = _mm_add_epi32(bias_vec1, dst_1);
                }
            }
            __m128 scale_vec0 = _mm_loadu_ps(scale_z + dc);
            __m128 scale_vec1 = _mm_loadu_ps(scale_z + dc + 4);
            __m128 dst_4x32_0 = _mm_cvtepi32_ps(bias_vec0);
            __m128 dst_4x32_1 = _mm_cvtepi32_ps(bias_vec1);
            dst_4x32_0        = _mm_mul_ps(dst_4x32_0, scale_vec0);
            dst_4x32_1        = _mm_mul_ps(dst_4x32_1, scale_vec1);

            // rounding to zero(val + (val >= 0.f ? 0.5f : -0.5f)) = rounding to nearest ties away from zero
            __m128 cmp_zero_0 = _mm_cmpge_ps(dst_4x32_0, zero_f32);
            __m128 cmp_zero_1 = _mm_cmpge_ps(dst_4x32_1, zero_f32);
            __m128 adjust_vec_0 = _mm_blendv_ps(sub_05, add_05, cmp_zero_0);
            __m128 adjust_vec_1 = _mm_blendv_ps(sub_05, add_05, cmp_zero_1);
            dst_4x32_0 = _mm_add_ps(dst_4x32_0, adjust_vec_0);
            dst_4x32_1 = _mm_add_ps(dst_4x32_1, adjust_vec_1);
            __m128i dst_8x8 = _mm_set_epi64(_mm_cvtps_pi8(dst_4x32_1), _mm_cvtps_pi8(dst_4x32_0));
            dst_8x8 = _mm_shuffle_epi32(dst_8x8, 8);
            _mm_storeu_si64(dst_x, dst_8x8);
        }
        for (; dc < dst_depth; dc += 4) {
            auto dst_x          = dst + dx * dst_depth + dc;
            const auto src_z    = src + dx * src_w_step + dc;
            int32_t dstInt32[4] = {0, 0, 0, 0};
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilate_y_step;
                const auto weight_y = weight + fy * fw * dst_depth + dc;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilate_x_step;
                    const auto weight_x = weight_y + dst_depth * fx;
                    for (long j = 0; j < 4; ++j) {
                        dstInt32[j] += (int32_t)src_x[j] * (int32_t)weight_x[j];
                    }
                }
            }

            for (long i = 0; i < 4; ++i) {
                dst_x[i] = float2int8(static_cast<float>(dstInt32[i] + bias_z[i + dc]) * scale_z[i + dc]);
            }
        }
    }
}

void X86ReluInt8(int8_t* dst, const int8_t* src, long len) {
    __m128i zero_i8 = _mm_setzero_si128();
    long idx = len - len % 16;

    for (long i = 0; i < idx; i += 16) {
        __m128i vec = _mm_loadu_si128((__m128i*)(src + i));
        _mm_storeu_si128((__m128i*)(dst + i), _mm_max_epi8(vec, zero_i8));
    }
    for (; idx < len; idx++) {
        dst[idx] = MAX(0, src[idx]);
    }
}

void X86Relu6Int8(int8_t* dst, const int8_t* src, const int8_t* relu6_max, long width, long dst_depth) {
    __m128i zero_i8 = _mm_setzero_si128();

    for (long dx = 0; dx < width; dx++) {
        auto src_dx = src + dx * dst_depth;
        auto dst_dx = dst + dx * dst_depth;

        long dc = 0;
        for (; dc + 15 < dst_depth; dc += 16) {
            __m128i src_vec   = _mm_loadu_si128((__m128i*)(src_dx + dc));
            __m128i relu6_vec = _mm_loadu_si128((__m128i*)(relu6_max + dc));
            _mm_storeu_si128((__m128i*)(dst_dx + dc), 
                _mm_max_epi8(zero_i8, _mm_min_epi8(src_vec, relu6_vec)));
        }
        for (; dc < dst_depth; dc++) {
            int8_t tmp = MIN(src_dx[dc], relu6_max[dc]);
            dst_dx[dc] = MAX(0, tmp);
        }
    }
}

}   // namespace TNN_NS
