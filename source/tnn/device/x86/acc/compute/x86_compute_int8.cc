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

// rounding to zero(val + (val >= 0.f ? 0.5f : -0.5f)) = rounding to nearest ties away from zero
#define DeclareRounding()                                             \
    __m128 zero_f32 = _mm_set1_ps(0.f);                               \
    __m128 add_05   = _mm_set1_ps(0.5f);                              \
    __m128 sub_05   = _mm_set1_ps(-0.5f);

#define F32X4TOI8X4(f32x4, dst)                                       \
    __m128 cmp_zero = _mm_cmpge_ps(f32x4, zero_f32);                  \
    __m128 adjust_vec = _mm_blendv_ps(sub_05, add_05, cmp_zero);      \
    f32x4 = _mm_add_ps(f32x4, adjust_vec);                            \
    __m128i dst_i32x4 = _mm_cvttps_epi32(f32x4);                      \
    __m128i dst_i16x4 = _mm_packs_epi32(dst_i32x4, dst_i32x4);        \
    __m128i dst_i8x4  = _mm_packs_epi16(dst_i16x4, dst_i16x4);        \
    int i8x4 = _mm_extract_epi32(dst_i8x4, 0);                        \
    *((int*)(dst)) = i8x4;

#define F32X8TOI8X8(f32x4_a, f32x4_b, dst)                            \
    __m128 cmp_zero_0 = _mm_cmpge_ps(f32x4_a, zero_f32);              \
    __m128 cmp_zero_1 = _mm_cmpge_ps(f32x4_b, zero_f32);              \
    __m128 adjust_vec_0 = _mm_blendv_ps(sub_05, add_05, cmp_zero_0);  \
    __m128 adjust_vec_1 = _mm_blendv_ps(sub_05, add_05, cmp_zero_1);  \
    f32x4_a = _mm_add_ps(f32x4_a, adjust_vec_0);                      \
    f32x4_b = _mm_add_ps(f32x4_b, adjust_vec_1);                      \
    __m128i dst_i32x4_a = _mm_cvttps_epi32(f32x4_a);                  \
    __m128i dst_i32x4_b = _mm_cvttps_epi32(f32x4_b);                  \
    __m128i dst_i16x8   = _mm_packs_epi32(dst_i32x4_a, dst_i32x4_b);  \
    __m128i dst_i8x8    = _mm_packs_epi16(dst_i16x8, dst_i16x8);      \
    _mm_storeu_si64(dst, dst_i8x8);

void X86GemmInt8Unit4x4(const int8_t* src, const int8_t* weight, int8_t* dst, long src_w_step, long dst_depth, long cdiv8,
                     const float* scale, const int32_t* bias, long relu, const int8_t* add_input,
                     const float* add_scale, const int8_t* relu6_max) {
    DeclareRounding();
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
        F32X4TOI8X4(dst_4x32, dst_x);
    }
}

static void DepthwiseI8K3Kernel(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z,
                                long src_y_step, long src_w_step, long dst_depth, const float* scale_z,
                                long dx, long dc) {
    DeclareRounding();
    __m128i zero_i8 = _mm_setzero_si128();

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

    F32X8TOI8X8(dst_4x32_0, dst_4x32_1, dst_x);
}

void DepthwiseI8K5Kernel(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z,
                         long src_y_step, long src_w_step, long dst_depth, const float* scale_z,
                         long dx, long dc) {
    DeclareRounding();
    __m128i zero_i8 = _mm_setzero_si128();

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

    F32X8TOI8X8(dst_4x32_0, dst_4x32_1, dst_x);
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
    DeclareRounding();
    __m128i zero_i8 = _mm_setzero_si128();
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

        F32X8TOI8X8(dst_4x32_0, dst_4x32_1, (dst + dc));
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
    DeclareRounding();
    __m128i zero_i8 = _mm_setzero_si128();

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

            F32X8TOI8X8(dst_4x32_0, dst_4x32_1, dst_x);
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

void X86MaxPoolingINT8(const int8_t* src, long iw, long ih, int8_t* dst, long ow, long oh, long c_r4, long kw, long kh,
                    long stride_w, long stride_h, long pad_w, long pad_h) {
    for (long oy = 0; oy < oh; ++oy) {
        for (long ox = 0; ox < ow; ++ox) {
            const long srcOriginX = ox * stride_w - pad_w;
            const long srcOriginY = oy * stride_h - pad_h;
            const long kxs        = MAX(0, -srcOriginX);
            const long kxe        = MIN(kw, iw - srcOriginX);
            const long kys        = MAX(0, -srcOriginY);
            const long kye        = MIN(kh, ih - srcOriginY);
            long oc               = 0;

            for (; oc + 15 < c_r4; oc += 16) {
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                __m128i max_reg    = _mm_set1_epi8(-127);
                // find kernel_w * kernel_h max value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;
                    for (; kx < kxe; kx++) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        max_reg                = _mm_max_epi8(max_reg, _mm_loadu_si128((__m128i*)srcPtrStart));
                    }
                }
                _mm_storeu_si128((__m128i*)dst_ptr, max_reg);
            }
            for (; oc + 7 < c_r4; oc += 8) {
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                __m128i max_reg    = _mm_set1_epi8(-127);
                // find kernel_w * kernel_h max value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;
                    for (; kx < kxe; kx++) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        max_reg                = _mm_max_epi8(max_reg, _mm_loadu_si64(srcPtrStart));
                    }
                }
                _mm_storeu_si64(dst_ptr, max_reg);
            }
            for (; oc < c_r4; oc += 4) {
                int8_t maxValue[4] = {-127, -127, -127, -127};
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                // find kernel_w * kernel_h max value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;
                    for (; kx < kxe; ++kx) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        for (long j = 0; j < 4; ++j) {
                            maxValue[j] = MAX(maxValue[j], srcPtrStart[j]);
                        }
                    }
                }
                // output
                *(int32_t*)dst_ptr = *(int32_t*)maxValue;
            }
        }
    }
}

void X86AvgPoolingINT8(const int8_t* src, long iw, long ih, int8_t* dst, long ow, long oh, long c_r4, long kw, long kh,
                    long stride_w, long stride_h, long pad_w, long pad_h) {
    for (long oy = 0; oy < oh; ++oy) {
        for (long ox = 0; ox < ow; ++ox) {
            const long srcOriginX   = ox * stride_w - pad_w;
            const long srcOriginY   = oy * stride_h - pad_h;
            const long kxs          = MAX(0, -srcOriginX);
            const long kxe          = MIN(kw, iw - srcOriginX);
            const long kys          = MAX(0, -srcOriginY);
            const long kye          = MIN(kh, ih - srcOriginY);
            const long kernel_count = (kxe - kxs) * (kye - kys);
            long oc                 = 0;

            int16_t sum[8];
            __m128 div_vec = _mm_set1_ps((float)kernel_count);
            for (; oc + 7 < c_r4; oc += 8) {
                __m128i avg_reg    = _mm_setzero_si128();
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                // find kernel_w * kernel_h avg value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;
                    for (; kx < kxe; kx++) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        __m128i cur_val = _mm_cvtepi8_epi16(_mm_loadu_si64(srcPtrStart));
                        avg_reg         = _mm_add_epi16(avg_reg, cur_val);
                    }
                }
                __m128 avg_reg_lo = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpacklo_epi64(avg_reg, avg_reg)));
                __m128 avg_reg_hi = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(avg_reg, avg_reg)));
                avg_reg_lo        = _mm_div_ps(avg_reg_lo, div_vec);
                avg_reg_hi        = _mm_div_ps(avg_reg_hi, div_vec);

                __m128i i32x8_a   = _mm_cvttps_epi32(avg_reg_lo);
                __m128i i32x8_b   = _mm_cvttps_epi32(avg_reg_hi);
                __m128i i16x8     = _mm_packs_epi32(i32x8_a, i32x8_b);
                __m128i i8x8      = _mm_packs_epi16(i16x8, i16x8);
                _mm_storeu_si64(dst_ptr, i8x8);
            }

            for (; oc < c_r4; oc += 4) {
                int16_t sum[4]     = {0, 0, 0, 0};
                const auto src_ptr = src + (srcOriginY * iw + srcOriginX) * c_r4 + oc;
                auto dst_ptr       = dst + (oy * ow + ox) * c_r4 + oc;
                // find kernel_w * kernel_h avg value
                for (long ky = kys; ky < kye; ++ky) {
                    const auto src_ptr_h = src_ptr + (ky * iw) * c_r4;
                    long kx              = kxs;

                    for (; kx < kxe; ++kx) {
                        const auto srcPtrStart = src_ptr_h + kx * c_r4;
                        for (long j = 0; j < 4; ++j) {
                            sum[j] += srcPtrStart[j];
                        }
                    }
                }
                // output
                for (long j = 0; j < 4; j++) {
                    dst_ptr[j] = static_cast<int8_t>(sum[j] / kernel_count);
                }
            }
        }
    }
}

/*
element add int8 func
*/
void X86MatrixAddInt8(int8_t* dst, const int8_t* A, const int8_t* B, float* dst_scale, const float* a_scale,
                   float* b_scale, long channel, long hw_size) {
    DeclareRounding();

    for (long hw = 0; hw < hw_size; hw++) {
        long c = 0;

        auto A_hw   = A + hw * channel;
        auto B_hw   = B + hw * channel;
        auto dst_hw = dst + hw * channel;
        for (; c < channel - 4; c += 8) {
            __m128 scale_a_neon0   = _mm_loadu_ps(a_scale + c);
            __m128 scale_a_neon1   = _mm_loadu_ps(a_scale + c + 4);
            __m128 scale_b_neon0   = _mm_loadu_ps(b_scale + c);
            __m128 scale_b_neon1   = _mm_loadu_ps(b_scale + c + 4);
            __m128 scale_dst_neon0 = _mm_loadu_ps(dst_scale + c);
            __m128 scale_dst_neon1 = _mm_loadu_ps(dst_scale + c + 4);

            __m128i aval       = _mm_cvtepi8_epi16(_mm_loadu_si64(A_hw + c));
            __m128i bval       = _mm_cvtepi8_epi16(_mm_loadu_si64(B_hw + c));
            __m128 a0          = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpacklo_epi64(aval, aval)));
            __m128 a1          = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(aval, aval)));
            __m128 b0          = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpacklo_epi64(bval, bval)));
            __m128 b1          = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(bval, bval)));
            __m128 mul0        = _mm_add_ps(_mm_mul_ps(a0, scale_a_neon0), _mm_mul_ps(b0, scale_b_neon0));
            __m128 mul1        = _mm_add_ps(_mm_mul_ps(a1, scale_a_neon1), _mm_mul_ps(b1, scale_b_neon1));
            mul0               = _mm_mul_ps(mul0, scale_dst_neon0);
            mul1               = _mm_mul_ps(mul1, scale_dst_neon1);

            F32X8TOI8X8(mul0, mul1, (dst_hw + c));
        }
        for (; c < channel; c++) {
            float aval  = A_hw[c] * a_scale[c] + B_hw[c] * b_scale[c];
            dst_hw[c] = float2int8(aval * dst_scale[c]);
        }
    }
}

void X86GemvInt8(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, const float* scale, long ic_r4,
              long oc_r4) {
    DeclareRounding();

    for (long dc = 0; dc < oc_r4; dc += 4) {
        __m128i acc0 = _mm_setzero_si128();
        __m128i acc1 = _mm_setzero_si128();
        __m128i acc2 = _mm_setzero_si128();
        __m128i acc3 = _mm_setzero_si128();
        auto weight_o_0 = weight + dc * ic_r4;
        auto weight_o_1 = weight_o_0 + ic_r4;
        auto weight_o_2 = weight_o_1 + ic_r4;
        auto weight_o_3 = weight_o_2 + ic_r4;

        long c         = 0;
        for (; c + 15 < ic_r4; c += 16) {
            __m128i a     = _mm_loadu_si128((__m128i*)(src + c));
            __m128i b0    = _mm_loadu_si128((__m128i*)(weight_o_0 + c));
            __m128i b1    = _mm_loadu_si128((__m128i*)(weight_o_1 + c));
            __m128i b2    = _mm_loadu_si128((__m128i*)(weight_o_2 + c));
            __m128i b3    = _mm_loadu_si128((__m128i*)(weight_o_3 + c));

            __m128i a_lo  = _mm_cvtepi8_epi16(_mm_unpacklo_epi64(a, a));
            __m128i a_hi  = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(a, a));
            __m128i b0_lo = _mm_cvtepi8_epi16(_mm_unpacklo_epi64(b0, b0));
            __m128i b0_hi = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(b0, b0));
            __m128i b1_lo = _mm_cvtepi8_epi16(_mm_unpacklo_epi64(b1, b1));
            __m128i b1_hi = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(b1, b1));
            __m128i b2_lo = _mm_cvtepi8_epi16(_mm_unpacklo_epi64(b2, b2));
            __m128i b2_hi = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(b2, b2));
            __m128i b3_lo = _mm_cvtepi8_epi16(_mm_unpacklo_epi64(b3, b3));
            __m128i b3_hi = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(b3, b3));

            acc0          = _mm_add_epi32(acc0, _mm_madd_epi16(a_lo, b0_lo));
            acc1          = _mm_add_epi32(acc1, _mm_madd_epi16(a_lo, b1_lo));
            acc2          = _mm_add_epi32(acc2, _mm_madd_epi16(a_lo, b2_lo));
            acc3          = _mm_add_epi32(acc3, _mm_madd_epi16(a_lo, b3_lo));
            acc0          = _mm_add_epi32(acc0, _mm_madd_epi16(a_hi, b0_hi));
            acc1          = _mm_add_epi32(acc1, _mm_madd_epi16(a_hi, b1_hi));
            acc2          = _mm_add_epi32(acc2, _mm_madd_epi16(a_hi, b2_hi));
            acc3          = _mm_add_epi32(acc3, _mm_madd_epi16(a_hi, b3_hi));
        }
        for (; c + 7 < ic_r4; c += 8) {
            __m128i a  = _mm_cvtepi8_epi16(_mm_loadu_si64(src + c));
            __m128i b0 = _mm_cvtepi8_epi16(_mm_loadu_si64(weight_o_0 + c));
            __m128i b1 = _mm_cvtepi8_epi16(_mm_loadu_si64(weight_o_1 + c));
            __m128i b2 = _mm_cvtepi8_epi16(_mm_loadu_si64(weight_o_2 + c));
            __m128i b3 = _mm_cvtepi8_epi16(_mm_loadu_si64(weight_o_3 + c));

            acc0       = _mm_add_epi32(acc0, _mm_madd_epi16(a, b0));
            acc1       = _mm_add_epi32(acc1, _mm_madd_epi16(a, b1));
            acc2       = _mm_add_epi32(acc2, _mm_madd_epi16(a, b2));
            acc3       = _mm_add_epi32(acc3, _mm_madd_epi16(a, b3));
        }
        for (; c < ic_r4; c += 4) {
            int a_4xi8  = *((int*)(src + c));
            int b0_4xi8 = *((int*)(weight_o_0 + c));
            int b1_4xi8 = *((int*)(weight_o_1 + c));
            int b2_4xi8 = *((int*)(weight_o_2 + c));
            int b3_4xi8 = *((int*)(weight_o_3 + c));

            __m128i a  = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(a_4xi8));
            __m128i b0 = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(b0_4xi8));
            __m128i b1 = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(b1_4xi8));
            __m128i b2 = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(b2_4xi8));
            __m128i b3 = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(b3_4xi8));

            acc0       = _mm_add_epi32(acc0, _mm_madd_epi16(a, b0));
            acc1       = _mm_add_epi32(acc1, _mm_madd_epi16(a, b1));
            acc2       = _mm_add_epi32(acc2, _mm_madd_epi16(a, b2));
            acc3       = _mm_add_epi32(acc3, _mm_madd_epi16(a, b3));
        }

        __m128i dst_4xi32 = _mm_hadd_epi32(_mm_hadd_epi32(acc0, acc1), _mm_hadd_epi32(acc2, acc3));
        __m128i bias_vec  = _mm_loadu_si128((__m128i*)(bias + dc));
        __m128 scale_vec  = _mm_loadu_ps(scale + dc);
        __m128 dst_4xf32  = _mm_cvtepi32_ps(_mm_add_epi32(dst_4xi32, bias_vec));
        dst_4xf32         = _mm_mul_ps(dst_4xf32, scale_vec);

        F32X4TOI8X4(dst_4xf32, (dst + dc));
    }
}

}   // namespace TNN_NS
