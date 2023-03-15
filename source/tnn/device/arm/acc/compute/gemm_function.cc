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

#include "tnn/device/arm/acc/compute/gemm_function.h"

#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

#include "tnn/device/arm/acc/compute/compute.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

#ifdef __aarch64__
#define GEMM_KERNEL_FLOAT GEMM_FLOAT_N8
#define GEMM_KERNEL_BFP16 GEMM_BFP16_N8
#else
#define GEMM_KERNEL_FLOAT GEMM_FLOAT_N4
#define GEMM_KERNEL_BFP16 GEMM_BFP16_N4
#endif

template <typename T>
void GEMM_FUNC(T *dst, const T *src, const float *weight, int src_depth_quad, int dst_step, int dst_depth_quad,
               int width, float *bias, long relu) {
    LOGE("TYPE NOT IMPLEMENT");
}

template <>
void GEMM_FUNC(bfp16_t *dst, const bfp16_t *src, const float *weight, int src_depth_quad, int dst_step,
               int dst_depth_quad, int width, float *bias, long relu) {
    GEMM_KERNEL_BFP16(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, width, bias, relu);
}

template <>
void GEMM_FUNC(float *dst, const float *src, const float *weight, int src_depth_quad, int dst_step, int dst_depth_quad,
               int width, float *bias, long relu) {
    GEMM_KERNEL_FLOAT(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, width, bias, relu);
}

void set_block_size(int &a_block, int &b_block, int l2_size, const int plane_num, const int oc_r4, const int ic_r4,
                    int byte_size) {
    const int l1cache = 32 * 1024 / byte_size;
    if (plane_num >= oc_r4) {
        // inner kernel also a first, safe in l1 cache
        a_block = MAX(l1cache / ic_r4 - ARM_SGEMM_TILE_N, 1);
        // b safe in l2 cache
        int l2_size_b = l2_size / ic_r4 - a_block;
        b_block       = MIN(l2_size_b, oc_r4);
    } else {
        if (plane_num < l2_size / ic_r4 - ARM_SGEMM_TILE_N) {
            a_block = plane_num;
        } else {
            a_block = MAX(l2_size / ic_r4 - ARM_SGEMM_TILE_N, 1);
        }
        b_block = ARM_SGEMM_TILE_N;
    }
    b_block = ROUND_UP(b_block, ARM_SGEMM_TILE_N);
    a_block = ROUND_UP(a_block, ARM_SGEMM_TILE_M);
}

static inline void repack_lane(float *src, float *dst) {
    Float4x4 q0 = Float4x4::ld4(src + 0);
    Float4x4 q4 = Float4x4::ld4(src + 16);
#ifdef __aarch64__
    Float4x4 q8 = Float4x4::ld4(src + 32);

    Float4x4::st1_lane(dst + 0, q0, 0);
    Float4x4::st1_lane(dst + 4, q4, 0);
    Float4x4::st1_lane(dst + 8, q8, 0);
    Float4x4::st1_lane(dst + 12, q0, 1);
    Float4x4::st1_lane(dst + 16, q4, 1);
    Float4x4::st1_lane(dst + 20, q8, 1);
    Float4x4::st1_lane(dst + 24, q0, 2);
    Float4x4::st1_lane(dst + 28, q4, 2);
    Float4x4::st1_lane(dst + 32, q8, 2);
    Float4x4::st1_lane(dst + 36, q0, 3);
    Float4x4::st1_lane(dst + 40, q4, 3);
    Float4x4::st1_lane(dst + 44, q8, 3);
#else
    Float4x4::st1_lane(dst + 0, q0, 0);
    Float4x4::st1_lane(dst + 4, q4, 0);
    Float4x4::st1_lane(dst + 8, q0, 1);
    Float4x4::st1_lane(dst + 12, q4, 1);
    Float4x4::st1_lane(dst + 16, q0, 2);
    Float4x4::st1_lane(dst + 20, q4, 2);
    Float4x4::st1_lane(dst + 24, q0, 3);
    Float4x4::st1_lane(dst + 28, q4, 3);
#endif
}

static inline void _repack_4(float *src, float *dst) {
    Float4x4 q0 = Float4x4::ld4(src + 0);
    Float4x4::st1_lane(dst + 0, q0, 0);
    Float4x4::st1_lane(dst + 4, q0, 1);
    Float4x4::st1_lane(dst + 8, q0, 2);
    Float4x4::st1_lane(dst + 12, q0, 3);
}

static inline void _repack_4(bfp16_t *src, bfp16_t *dst) {
    int16_t *src_b = reinterpret_cast<int16_t *>(src);
    int16_t *dst_b = reinterpret_cast<int16_t *>(dst);
    Short4x4 q0    = Short4x4::ld4(src_b + 0);
    Short4x4::st1_lane(dst_b + 0, q0, 0);
    Short4x4::st1_lane(dst_b + 4, q0, 1);
    Short4x4::st1_lane(dst_b + 8, q0, 2);
    Short4x4::st1_lane(dst_b + 12, q0, 3);
}

static inline void repack_lane(bfp16_t *src, bfp16_t *dst) {
    int16_t *src_b = reinterpret_cast<int16_t *>(src);
    int16_t *dst_b = reinterpret_cast<int16_t *>(dst);
    Short4x4 q0    = Short4x4::ld4(src_b + 0);
    Short4x4 q4    = Short4x4::ld4(src_b + 16);
#ifdef __aarch64__
    Short4x4 q8 = Short4x4::ld4(src_b + 32);

    Short4x4::st1_lane(dst_b + 0, q0, 0);
    Short4x4::st1_lane(dst_b + 4, q4, 0);
    Short4x4::st1_lane(dst_b + 8, q8, 0);
    Short4x4::st1_lane(dst_b + 12, q0, 1);
    Short4x4::st1_lane(dst_b + 16, q4, 1);
    Short4x4::st1_lane(dst_b + 20, q8, 1);
    Short4x4::st1_lane(dst_b + 24, q0, 2);
    Short4x4::st1_lane(dst_b + 28, q4, 2);
    Short4x4::st1_lane(dst_b + 32, q8, 2);
    Short4x4::st1_lane(dst_b + 36, q0, 3);
    Short4x4::st1_lane(dst_b + 40, q4, 3);
    Short4x4::st1_lane(dst_b + 44, q8, 3);
#else
    Short4x4::st1_lane(dst_b + 0, q0, 0);
    Short4x4::st1_lane(dst_b + 4, q4, 0);
    Short4x4::st1_lane(dst_b + 8, q0, 1);
    Short4x4::st1_lane(dst_b + 12, q4, 1);
    Short4x4::st1_lane(dst_b + 16, q0, 2);
    Short4x4::st1_lane(dst_b + 20, q4, 2);
    Short4x4::st1_lane(dst_b + 24, q0, 3);
    Short4x4::st1_lane(dst_b + 28, q4, 3);
#endif
}

/*
armv8: repack12 x n(n>=0) + repack4 x n(0<=n<=2) + memcpy
armv8: repack8  x n(n>=0) + repack4 x n(0<=n<=1) + memcpy
*/
template <typename T>
void load_repack_A(T *dst, T *src, int width, int src_z_step, int ic4) {
    int loop   = width / ARM_SGEMM_TILE_M;
    int remain = width % ARM_SGEMM_TILE_M;

    // OMP_PARALLEL_FOR_
    for (int db = 0; db <= loop; db++) {
        auto src_b = src + db * ARM_SGEMM_TILE_M * 4;
        auto dst_b = dst + db * ARM_SGEMM_TILE_M * 4 * ic4;
        int len    = (db < loop) ? ARM_SGEMM_TILE_M : remain;
        if (len == ARM_SGEMM_TILE_M) {
            for (int c_i = 0; c_i < ic4; c_i++) {
                auto src_z = src_b + c_i * src_z_step;
                auto dst_z = dst_b + c_i * ARM_SGEMM_TILE_M * 4;
                repack_lane(src_z, dst_z);
            }
        } else if (len > 0) {
            for (int c_i = 0; c_i < ic4; c_i++) {
                auto src_z = src_b + c_i * src_z_step;
                auto dst_z = dst_b + c_i * len * 4;
                memcpy(dst_z, src_z, remain * 4 * sizeof(T));
                for (int b_i = 0; b_i < remain / 4; b_i++) {
                    auto src_r = src_z + b_i * 4 * 4;
                    auto dst_r = dst_z + b_i * 4 * 4;
                    _repack_4(src_r, dst_r);
                }
            }
        }
    }
}

template <typename T>
void sgemm_repack_lhs(T *dst, T *src, float *weight, int ic4, int oc4, int plane_num, int dst_z_step, int a_block,
                      int b_block, T *work_space, float *bias, int act_type, bool fast_post) {
    int loop                 = plane_num / a_block;
    int remain               = plane_num % a_block;
    int workspace_per_thread = a_block * ic4 * 4;
    int do_relu              = act_type == 1 || act_type == 2;

    OMP_PARALLEL_FOR_
    for (int db = 0; db <= loop; db++) {
        int thread_id = OMP_TID_;
        auto dst_b    = work_space + thread_id * workspace_per_thread;
        auto src_b    = src + db * a_block * 4;
        auto width    = (db < loop) ? a_block : remain;
        auto x_loop   = width / ARM_SGEMM_TILE_M;
        auto x_remain = width % ARM_SGEMM_TILE_M;

        load_repack_A(dst_b, src_b, width, plane_num * 4, ic4);

        auto weight_z_step = ic4 * b_block * 4;

        for (int c_o = 0; c_o < UP_DIV(oc4 * 4, b_block); c_o++) {
            /*
            a_block is much smaller in sgemm_lhs than that in sgemm_rhs
            same process with sgemm_lhs, but we load weight repeatedly
            */
            auto calc_b_block = MIN(b_block, oc4 * 4 - c_o * b_block);
            auto weight_ptr   = weight + c_o * weight_z_step;
            auto output_ptr   = dst + c_o * plane_num * b_block + db * a_block * 4;
            for (int x_i = 0; x_i <= x_loop; x_i++) {
                auto x_width = (x_i < x_loop) ? ARM_SGEMM_TILE_M : x_remain;
                GEMM_FUNC(output_ptr + x_i * ARM_SGEMM_TILE_M * 4, dst_b + x_i * ARM_SGEMM_TILE_M * ic4 * 4, weight_ptr,
                          ic4, dst_z_step, calc_b_block / 4, x_width, bias + c_o * b_block, do_relu);
            }
        }
    }

    // only bias + relu6 here, bias and bias + relu has been fused to gemm kernel
    if (act_type == 2)
        PostClap<T>(dst, plane_num * oc4, 6);
    else if (act_type == 0x0100) {
        if (fast_post)
            PostAddBiasSwish<T, float, true>(dst, nullptr, plane_num, oc4);
        else
            PostAddBiasSwish<T, float, false>(dst, nullptr, plane_num, oc4);
    }
}

template void sgemm_repack_lhs(float *dst, float *src, float *weight, int ic4, int oc4, int plane_num, int dst_z_step,
                               int a_block, int b_block, float *work_space, float *bias, int act_type, bool fast_post);

template void sgemm_repack_lhs(bfp16_t *dst, bfp16_t *src, float *weight, int ic4, int oc4, int plane_num,
                               int dst_z_step, int a_block, int b_block, bfp16_t *work_space, float *bias, int act_type,
                               bool fast_post);

template <typename T>
void sgemm_repack_rhs(T *dst, T *src, float *weight, int ic4, int oc4, int plane_num, int dst_z_step, int a_block,
                      int b_block, T *work_space, float *bias, int act_type, bool fast_post) {
    int loop    = plane_num / a_block;
    int remain  = plane_num % a_block;
    int do_relu = act_type == 1 || act_type == 2;

    for (int db = 0; db <= loop; db++) {
        auto src_b    = src + db * a_block * 4;
        auto dst_b    = work_space;
        auto width    = (db < loop) ? a_block : remain;
        auto x_loop   = width / ARM_SGEMM_TILE_M;
        auto x_remain = width % ARM_SGEMM_TILE_M;

        load_repack_A(dst_b, src_b, width, plane_num * 4, ic4);

        auto weight_z_step = ic4 * b_block * 4;

        OMP_PARALLEL_FOR_
        for (int c_o = 0; c_o < UP_DIV(oc4 * 4, b_block); c_o++) {
            /*
            a_block is much greater in sgemm_rhs than that in sgemm_lhs
            same process with sgemm_lhs, but we load data repeatedly
            */
            auto output_ptr   = dst + c_o * plane_num * b_block + db * a_block * 4;
            auto calc_b_block = MIN(b_block, oc4 * 4 - c_o * b_block);
            auto weight_ptr   = weight + c_o * weight_z_step;
            for (int x_i = 0; x_i <= x_loop; x_i++) {
                auto x_width = (x_i < x_loop) ? ARM_SGEMM_TILE_M : x_remain;
                GEMM_FUNC(output_ptr + x_i * ARM_SGEMM_TILE_M * 4, dst_b + x_i * ARM_SGEMM_TILE_M * ic4 * 4, weight_ptr,
                          ic4, dst_z_step, calc_b_block / 4, x_width, bias + c_o * b_block, do_relu);
            }
        }
    }

    // only bias + relu6 here, bias and bias + relu has been fused to gemm kernel
    if (act_type == 2)
        PostClap<T>(dst, plane_num * oc4, 6);
    else if (act_type == 0x0100) {
        if (fast_post)
            PostAddBiasSwish<T, float, true>(dst, nullptr, plane_num, oc4);
        else
            PostAddBiasSwish<T, float, false>(dst, nullptr, plane_num, oc4);
    }
}

template void sgemm_repack_rhs(float *dst, float *src, float *weight, int ic4, int oc4, int plane_num, int dst_z_step,
                               int a_block, int b_block, float *work_space, float *bias, int act_type, bool fast_post);

template void sgemm_repack_rhs(bfp16_t *dst, bfp16_t *src, float *weight, int ic4, int oc4, int plane_num,
                               int dst_z_step, int a_block, int b_block, bfp16_t *work_space, float *bias, int act_type,
                               bool fast_post);

template <int mr, int nr, typename T>
void NaiveKernel(int m, int n, int k, const T *sa, const T *sb, T *sc, int ldc) {
    const T *a = sa;
    const T *b = sb;
    T *c       = sc;
    for (int i = 0; i < m - mr + 1; i += mr) {
        for (int j = 0; j < n - nr + 1; j += nr) {
            for (int p = 0; p < k; ++p) {
                for (int ir = 0; ir < mr; ++ir) {
                    for (int jr = 0; jr < nr; ++jr) {
                        c[ir * ldc + jr] += a[ir] * b[jr];
                    }
                }
                a += mr;
                b += nr;
            }
            c += nr;
            a -= mr * k;
        }
        int remain = n % nr;
        if (remain) {
            for (int p = 0; p < k; ++p) {
                for (int ir = 0; ir < mr; ++ir) {
                    for (int jr = 0; jr < remain; ++jr) {
                        c[ir * ldc + jr] += a[ir] * b[jr];
                    }
                }
                a += mr;
                b += nr;
            }
            a -= mr * k;
        }
        sc += ldc * mr;
        c = sc;
        a += mr * k;
        b = sb;
    }
}

#if TNN_ARM82
template void NaiveKernel<8, 16, fp16_t>(int m, int n, int k, const fp16_t *sa, const fp16_t *sb, fp16_t *sc, int ldc);
template void NaiveKernel<4, 16, fp16_t>(int m, int n, int k, const fp16_t *sa, const fp16_t *sb, fp16_t *sc, int ldc);
template void NaiveKernel<1, 16, fp16_t>(int m, int n, int k, const fp16_t *sa, const fp16_t *sb, fp16_t *sc, int ldc);
#endif  // TNN_ARM82

void Kernel_12x8(int m, int n, int k, const float *sa, const float *sb, float *sc, int ldc) {
#if defined(TNN_USE_NEON) && defined(__aarch64__)
    for (int i = 0; i < m - 11; i += 12) {
        const float *ar = sa + i * k;
        const float *br = sb;
        float *cr       = sc + i * ldc;
        OMP_PARALLEL_FOR_
        for (int j = 0; j < n - 7; j += 8) {
            const float *a     = ar;
            const float *b     = br + j * k;
            float *c           = cr + j;
            int64_t ldc_offset = ldc * sizeof(float) - 16;
            int64_t k_64       = k;

                #define INIT12x8                        \
                "   mov x9,        %2               \n" \
                "   ld1 {v8.4s},  [x9], #16         \n" \
                "   ld1 {v20.4s}, [x9], %3          \n" \
                "   ld1 {v9.4s},  [x9], #16         \n" \
                "   ld1 {v21.4s}, [x9], %3          \n" \
                "   ld1 {v10.4s}, [x9], #16         \n" \
                "   ld1 {v22.4s}, [x9], %3          \n" \
                "   ld1 {v11.4s}, [x9], #16         \n" \
                "   ld1 {v23.4s}, [x9], %3          \n" \
                "   ld1 {v12.4s}, [x9], #16         \n" \
                "   ld1 {v24.4s}, [x9], %3          \n" \
                "   ld1 {v13.4s}, [x9], #16         \n" \
                "   ld1 {v25.4s}, [x9], %3          \n" \
                "   ld1 {v14.4s}, [x9], #16         \n" \
                "   ld1 {v26.4s}, [x9], %3          \n" \
                "   ld1 {v15.4s}, [x9], #16         \n" \
                "   ld1 {v27.4s}, [x9], %3          \n" \
                "   ld1 {v16.4s}, [x9], #16         \n" \
                "   ld1 {v28.4s}, [x9], %3          \n" \
                "   ld1 {v17.4s}, [x9], #16         \n" \
                "   ld1 {v29.4s}, [x9], %3          \n" \
                "   ld1 {v18.4s}, [x9], #16         \n" \
                "   ld1 {v30.4s}, [x9], %3          \n" \
                "   ld1 {v19.4s}, [x9], #16         \n" \
                "   ld1 {v31.4s}, [x9]              \n"

                #define SAVE12x8                        \
                "   mov x9,        %2               \n" \
                "   st1 {v8.4s},  [x9], #16         \n" \
                "   st1 {v20.4s}, [x9], %3          \n" \
                "   st1 {v9.4s},  [x9], #16         \n" \
                "   st1 {v21.4s}, [x9], %3          \n" \
                "   st1 {v10.4s}, [x9], #16         \n" \
                "   st1 {v22.4s}, [x9], %3          \n" \
                "   st1 {v11.4s}, [x9], #16         \n" \
                "   st1 {v23.4s}, [x9], %3          \n" \
                "   st1 {v12.4s}, [x9], #16         \n" \
                "   st1 {v24.4s}, [x9], %3          \n" \
                "   st1 {v13.4s}, [x9], #16         \n" \
                "   st1 {v25.4s}, [x9], %3          \n" \
                "   st1 {v14.4s}, [x9], #16         \n" \
                "   st1 {v26.4s}, [x9], %3          \n" \
                "   st1 {v15.4s}, [x9], #16         \n" \
                "   st1 {v27.4s}, [x9], %3          \n" \
                "   st1 {v16.4s}, [x9], #16         \n" \
                "   st1 {v28.4s}, [x9], %3          \n" \
                "   st1 {v17.4s}, [x9], #16         \n" \
                "   st1 {v29.4s}, [x9], %3          \n" \
                "   st1 {v18.4s}, [x9], #16         \n" \
                "   st1 {v30.4s}, [x9], %3          \n" \
                "   st1 {v19.4s}, [x9], #16         \n" \
                "   st1 {v31.4s}, [x9]              \n"

            asm volatile(
                "                                   \n"
                "   ld1 {v0.4s}, [%0], #16          \n"
                "   ld1 {v2.4s}, [%1], #16          \n"
                INIT12x8
                "mov x8,%4                          \n"
                "0:                                 \n"

                "   fmla v8.4s , v0.4s, v2.s[0]     \n"
                "   ld1 {v3.4s}, [%1], #16          \n"
                "   fmla v9.4s , v0.4s, v2.s[1]     \n"
                "   fmla v10.4s, v0.4s, v2.s[2]     \n"
                "   ld1 {v4.4s}, [%1], #16          \n"
                "   fmla v11.4s, v0.4s, v2.s[3]     \n"

                "   fmla v12.4s, v0.4s, v3.s[0]     \n"
                "   ld1 {v1.4s}, [%0], #16          \n"
                "   fmla v13.4s, v0.4s, v3.s[1]     \n"
                "   fmla v14.4s, v0.4s, v3.s[2]     \n"
                "   prfm pldl1keep, [%1, #64]       \n"
                "   fmla v15.4s, v0.4s, v3.s[3]     \n"

                "   fmla v16.4s, v0.4s, v4.s[0]     \n"
                "   prfm pldl1keep, [%0, #64]       \n"
                "   fmla v17.4s, v0.4s, v4.s[1]     \n"
                "   fmla v18.4s, v0.4s, v4.s[2]     \n"
                "   prfm pldl1keep, [%1, #128]      \n"
                "   fmla v19.4s, v0.4s, v4.s[3]     \n"

                "   fmla v20.4s, v1.4s, v2.s[0]     \n"
                "   ld1 {v0.4s}, [%0], #16          \n"
                "   fmla v21.4s, v1.4s, v2.s[1]     \n"
                "   fmla v22.4s, v1.4s, v2.s[2]     \n"
                "   prfm pldl1keep, [%0, #128]      \n"
                "   fmla v23.4s, v1.4s, v2.s[3]     \n"
                "   subs x8, x8, #1                 \n"

                "   fmla v24.4s, v1.4s, v3.s[0]     \n"
                "   ld1 {v2.4s}, [%1], #16          \n"
                "   fmla v25.4s, v1.4s, v3.s[1]     \n"
                "   fmla v26.4s, v1.4s, v3.s[2]     \n"
                "   prfm pldl1keep, [%1, #192]      \n"
                "   fmla v27.4s, v1.4s, v3.s[3]     \n"

                "   fmla v28.4s, v1.4s, v4.s[0]     \n"
                "   prfm pldl1keep, [%0, #192]      \n"
                "   fmla v29.4s, v1.4s, v4.s[1]     \n"
                "   fmla v30.4s, v1.4s, v4.s[2]     \n"
                "   prfm pldl1keep, [%1, #256]      \n"
                "   fmla v31.4s, v1.4s, v4.s[3]     \n"
                "   bne 0b                          \n"
                SAVE12x8
                "                                   \n"
                : "=r"(b), "=r"(a), "=r"(c), "=r"(ldc_offset), "=r"(k_64)
                : "0"(b), "1"(a), "2"(c), "3"(ldc_offset), "4"(k_64)
                : "memory", "cc", "x8", "x9", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12", "v13",
                  "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                  "v28", "v29", "v30", "v31");
        }
        int remain = n % 8;
        if (remain) {
            const float *a     = ar;
            const float *b     = br + (n / 8) * 8 * k;
            float *c           = cr + (n / 8) * 8;
            float32x4_t c0[12] = {vdupq_n_f32(0)};
            float32x4_t c1[12] = {vdupq_n_f32(0)};
            float32x4_t b0, b1, av, a0, a1, a2, a3;
            for (int kk = 0; kk < k; ++kk) {
                b0    = vld1q_f32(b);
                b1    = vld1q_f32(b + 4);
                av    = vld1q_f32(a);
                a0    = vdupq_n_f32(av[0]);
                a1    = vdupq_n_f32(av[1]);
                a2    = vdupq_n_f32(av[2]);
                a3    = vdupq_n_f32(av[3]);
                c0[0] = vmlaq_f32(c0[0], a0, b0);
                c1[0] = vmlaq_f32(c1[0], a0, b1);
                c0[1] = vmlaq_f32(c0[1], a1, b0);
                c1[1] = vmlaq_f32(c1[1], a1, b1);
                c0[2] = vmlaq_f32(c0[2], a2, b0);
                c1[2] = vmlaq_f32(c1[2], a2, b1);
                c0[3] = vmlaq_f32(c0[3], a3, b0);
                c1[3] = vmlaq_f32(c1[3], a3, b1);

                av    = vld1q_f32(a + 4);
                a0    = vdupq_n_f32(av[0]);
                a1    = vdupq_n_f32(av[1]);
                a2    = vdupq_n_f32(av[2]);
                a3    = vdupq_n_f32(av[3]);
                c0[4] = vmlaq_f32(c0[4], a0, b0);
                c1[4] = vmlaq_f32(c1[4], a0, b1);
                c0[5] = vmlaq_f32(c0[5], a1, b0);
                c1[5] = vmlaq_f32(c1[5], a1, b1);
                c0[6] = vmlaq_f32(c0[6], a2, b0);
                c1[6] = vmlaq_f32(c1[6], a2, b1);
                c0[7] = vmlaq_f32(c0[7], a3, b0);
                c1[7] = vmlaq_f32(c1[7], a3, b1);

                av     = vld1q_f32(a + 8);
                a0     = vdupq_n_f32(av[0]);
                a1     = vdupq_n_f32(av[1]);
                a2     = vdupq_n_f32(av[2]);
                a3     = vdupq_n_f32(av[3]);
                c0[8]  = vmlaq_f32(c0[8], a0, b0);
                c1[8]  = vmlaq_f32(c1[8], a0, b1);
                c0[9]  = vmlaq_f32(c0[9], a1, b0);
                c1[9]  = vmlaq_f32(c1[9], a1, b1);
                c0[10] = vmlaq_f32(c0[10], a2, b0);
                c1[10] = vmlaq_f32(c1[10], a2, b1);
                c0[11] = vmlaq_f32(c0[11], a3, b0);
                c1[11] = vmlaq_f32(c1[11], a3, b1);

                b += 8;
                a += 12;
            }
            for (int ms = 0; ms < 12; ++ms) {
                for (int rr = 0; rr < remain; ++rr) {
                    c[rr] += rr < 4 ? c0[ms][rr] : c1[ms][rr - 4];
                }
                c += ldc;
            }
        }
    }
#else
    return NaiveKernel<12, 8>(m, n, k, sa, sb, sc, ldc);
#endif
}

void Kernel_4x8(int m, int n, int k, const float *sa, const float *sb, float *sc, int ldc) {
#ifdef TNN_USE_NEON
    for (int i = 0; i < m - 3; i += 4) {
        const float *ar = sa + i * k;
        const float *br = sb;
        float *cr       = sc + i * ldc;
        OMP_PARALLEL_FOR_
        for (int j = 0; j < n - 7; j += 8) {
            const float *a = ar;
            const float *b = br + j * k;
            float *c       = cr + j;
#ifdef __aarch64__
            int64_t ldc_offset = ldc * sizeof(float) - 16;
            int64_t k_64       = k;
                #define INIT4x8                         \
                "   mov x9,        %2               \n" \
                "   ld1 {v8.4s},  [x9], #16         \n" \
                "   ld1 {v20.4s}, [x9], %3          \n" \
                "   ld1 {v9.4s},  [x9], #16         \n" \
                "   ld1 {v21.4s}, [x9], %3          \n" \
                "   ld1 {v10.4s}, [x9], #16         \n" \
                "   ld1 {v22.4s}, [x9], %3          \n" \
                "   ld1 {v11.4s}, [x9], #16         \n" \
                "   ld1 {v23.4s}, [x9], %3          \n"

                #define SAVE4x8                         \
                "   mov x9,        %2               \n" \
                "   st1 {v8.4s},  [x9], #16         \n" \
                "   st1 {v20.4s}, [x9], %3          \n" \
                "   st1 {v9.4s},  [x9], #16         \n" \
                "   st1 {v21.4s}, [x9], %3          \n" \
                "   st1 {v10.4s}, [x9], #16         \n" \
                "   st1 {v22.4s}, [x9], %3          \n" \
                "   st1 {v11.4s}, [x9], #16         \n" \
                "   st1 {v23.4s}, [x9], %3          \n"

            asm volatile(
                "   ld1 {v0.4s}, [%0], #16          \n"
                "   ld1 {v2.4s}, [%1], #16          \n"
                INIT4x8
                "mov x8,%4                          \n"
                "0:                                 \n"

                "   fmla v8.4s , v0.4s, v2.s[0]     \n"
                "   fmla v9.4s , v0.4s, v2.s[1]     \n"
                "   fmla v10.4s, v0.4s, v2.s[2]     \n"
                "   fmla v11.4s, v0.4s, v2.s[3]     \n"

                "   ld1 {v1.4s}, [%0], #16          \n"
                "   prfm pldl1keep, [%1, #64]       \n"
                "   prfm pldl1keep, [%0, #64]       \n"

                "   fmla v20.4s, v1.4s, v2.s[0]     \n"
                "   ld1 {v0.4s}, [%0], #16          \n"
                "   fmla v21.4s, v1.4s, v2.s[1]     \n"
                "   fmla v22.4s, v1.4s, v2.s[2]     \n"
                "   prfm pldl1keep, [%0, #128]      \n"
                "   fmla v23.4s, v1.4s, v2.s[3]     \n"
                "   subs x8, x8, #1                 \n"
                "   ld1 {v2.4s}, [%1], #16          \n"
                "   bne 0b                          \n"
                SAVE4x8
                "                                   \n"
                : "=r"(b), "=r"(a), "=r"(c), "=r"(ldc_offset), "=r"(k_64)
                : "0"(b), "1"(a), "2"(c), "3"(ldc_offset), "4"(k_64)
                : "memory", "cc", "x8", "x9", "v0", "v1", "v2", "v8", "v9", "v10", "v11", "v20", "v21", "v22", "v23");
#else
            int ldc_offset = ldc * sizeof(float) - 16;
            #define INIT4x8                             \
                "   mov r9,        %2               \n" \
                "   vld1.f32 {d16,d17},  [r9]!      \n" \
                "   vld1.f32 {d18,d19},  [r9]       \n" \
                "   add      r9,   r9, %3           \n" \
                "   vld1.f32 {d20,d21}, [r9]!       \n" \
                "   vld1.f32 {d22,d23}, [r9]        \n" \
                "   add      r9,   r9, %3           \n" \
                "   vld1.f32 {d24,d25}, [r9]!       \n" \
                "   vld1.f32 {d26,d27}, [r9]        \n" \
                "   add      r9,   r9, %3           \n" \
                "   vld1.f32 {d28,d29}, [r9]!       \n" \
                "   vld1.f32 {d30,d31}, [r9]        \n" 

            #define SAVE4x8                             \
                "   mov r9,        %2               \n" \
                "   vst1.f32 {d16,d17},  [r9]!      \n" \
                "   vst1.f32 {d18,d19},  [r9]       \n" \
                "   add      r9,   r9, %3           \n" \
                "   vst1.f32 {d20,d21}, [r9]!       \n" \
                "   vst1.f32 {d22,d23}, [r9]        \n" \
                "   add      r9,   r9, %3           \n" \
                "   vst1.f32 {d24,d25}, [r9]!       \n" \
                "   vst1.f32 {d26,d27}, [r9]        \n" \
                "   add      r9,   r9, %3           \n" \
                "   vst1.f32 {d28,d29}, [r9]!       \n" \
                "   vst1.f32 {d30,d31}, [r9]        \n"

            asm volatile(
                "   vld1.f32 {d0,d1},  [%0]!        \n"
                "   vld1.f32 {d4,d5},  [%1]!        \n"
                INIT4x8
                "mov r8,%4                          \n"
                "0:                                 \n"
                "   vmla.f32 q8,  q0, d4[0]         \n"
                "   vld1.f32 {d2,d3},  [%0]!        \n"
                "   vmla.f32 q10, q0, d4[1]         \n"
                "   vmla.f32 q12, q0, d5[0]         \n"
                "   vmla.f32 q14, q0, d5[1]         \n"
                "   subs r8, r8, #1                 \n"

                "   vmla.f32 q9,  q1, d4[0]         \n"
                "   vmla.f32 q11, q1, d4[1]         \n"
                "   vmla.f32 q13, q1, d5[0]         \n"
                "   vld1.f32 {d0,d1},  [%0]!        \n"
                "   vmla.f32 q15, q1, d5[1]         \n"
                "   vld1.f32 {d4,d5},  [%1]!        \n"
                "   bne 0b                          \n"
                SAVE4x8
                "                                   \n"
                : "=r"(b), "=r"(a), "=r"(c), "=r"(ldc_offset), "=r"(k)
                : "0"(b), "1"(a), "2"(c), "3"(ldc_offset), "4"(k)
                : "memory", "cc", "r8", "r9", "q0", "q1", "q2", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif  // __aarch64__
        }
        int remain = n % 8;
        if (remain) {
            const float *a    = ar;
            const float *b    = br + (n / 8) * 8 * k;
            float *c          = cr + (n / 8) * 8;
            float32x4_t c0[4] = {vdupq_n_f32(0)};
            float32x4_t c1[4] = {vdupq_n_f32(0)};
            float32x4_t b0, b1, av, a0, a1, a2, a3;
            for (int kk = 0; kk < k; ++kk) {
                b0    = vld1q_f32(b);
                b1    = vld1q_f32(b + 4);
                av    = vld1q_f32(a);
                a0    = vdupq_n_f32(vgetq_lane_f32(av, 0));
                a1    = vdupq_n_f32(vgetq_lane_f32(av, 1));
                a2    = vdupq_n_f32(vgetq_lane_f32(av, 2));
                a3    = vdupq_n_f32(vgetq_lane_f32(av, 3));
                c0[0] = vmlaq_f32(c0[0], a0, b0);
                c1[0] = vmlaq_f32(c1[0], a0, b1);
                c0[1] = vmlaq_f32(c0[1], a1, b0);
                c1[1] = vmlaq_f32(c1[1], a1, b1);
                c0[2] = vmlaq_f32(c0[2], a2, b0);
                c1[2] = vmlaq_f32(c1[2], a2, b1);
                c0[3] = vmlaq_f32(c0[3], a3, b0);
                c1[3] = vmlaq_f32(c1[3], a3, b1);
                b += 8;
                a += 4;
            }
            for (int ms = 0; ms < 4; ++ms) {
                Float4 c0_ms(c0[ms]);
                Float4 c1_ms(c1[ms]);
                for (int rr = 0; rr < remain; ++rr) {
                    c[rr] += rr < 4 ? c0_ms[rr] : c1_ms[rr - 4];
                }
                c += ldc;
            }
        }
    }
#else
    return NaiveKernel<4, 8>(m, n, k, sa, sb, sc, ldc);
#endif  // TNN_USE_NEON
}

void Kernel_1x8(int m, int n, int k, const float *sa, const float *sb, float *sc, int ldc) {
#ifdef TNN_USE_NEON
    for (int i = 0; i < m; ++i) {
        const float *ar = sa + i * k;
        const float *br = sb;
        float *cr       = sc + i * ldc;
        OMP_PARALLEL_FOR_
        for (int j = 0; j < n - 7; j += 8) {
            const float *a = ar;
            const float *b = br + j * k;
            float *c       = cr + j;
#ifdef __aarch64__
            int64_t ldc_offset = ldc * sizeof(float) - 16;
            int64_t k_64       = k;

            #define INIT1x8                             \
                "   mov x9,        %2               \n" \
                "   ld1 {v8.4s},  [x9], #16         \n" \
                "   ld1 {v20.4s}, [x9], %3          \n" \
                "   movi v9.4s,    #0               \n" \
                "   movi v21.4s,   #0               \n"
            #define SAVE1x8                             \
                "   mov x9,        %2               \n" \
                "   fadd v8.4s,  v8.4s,  v9.4s      \n" \
                "   fadd v20.4s, v20.4s, v21.4s     \n" \
                "   st1 {v8.4s},  [x9], #16         \n" \
                "   st1 {v20.4s}, [x9], %3          \n" \

            asm volatile(
                "   ld1 {v0.4s}, [%0], #16          \n"
                "   ld1 {v2.4s}, [%1], #16          \n"
                INIT1x8
                "mov x8,%4                          \n"
                "0:                                 \n"
                "   subs x9, x8, #4                 \n"
                "   blt 1f                          \n"
                "   ld1 {v1.4s}, [%0], #16          \n"
                "   fmla v8.4s , v0.4s, v2.s[0]     \n"
                "   ld1 {v3.4s}, [%0], #16          \n"
                "   fmla v20.4s, v1.4s, v2.s[0]     \n"
                "   ld1 {v4.4s}, [%0], #16          \n"
                "   fmla v9.4s , v3.4s, v2.s[1]     \n"
                "   ld1 {v0.4s}, [%0], #16          \n"
                "   fmla v21.4s, v4.4s, v2.s[1]     \n"
                "   subs x8, x8, #4                 \n"

                "   ld1 {v1.4s}, [%0], #16          \n"
                "   fmla v8.4s , v0.4s, v2.s[2]     \n"
                "   ld1 {v3.4s}, [%0], #16          \n"
                "   fmla v20.4s, v1.4s, v2.s[2]     \n"
                "   ld1 {v4.4s}, [%0], #16          \n"
                "   fmla v9.4s , v3.4s, v2.s[3]     \n"
                "   ld1 {v0.4s}, [%0], #16          \n"
                "   fmla v21.4s, v4.4s, v2.s[3]     \n"
                "   ld1 {v2.4s}, [%1], #16          \n"
                "   bgt 0b                          \n"
                "1:                                 \n"
                "   subs x8, x8, #1                 \n"
                "   ld1 {v1.4s}, [%0], #16          \n"
                "   blt 2f                          \n"
                "   fmla v8.4s , v0.4s, v2.s[0]     \n"
                "   fmla v20.4s, v1.4s, v2.s[0]     \n"
                "   sub %1, %1, #12                 \n"
                "   ld1 {v0.4s}, [%0], #16          \n"
                "   ld1 {v2.4s}, [%1], #16          \n"
                "   bne 1b                          \n"
                "2:                                 \n"
                    SAVE1x8
                "                                   \n"
                : "=r"(b), "=r"(a), "=r"(c), "=r"(ldc_offset), "=r"(k_64)
                : "0"(b), "1"(a), "2"(c), "3"(ldc_offset), "4"(k_64)
                : "memory", "cc", "x8", "x9", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v20", "v21");

            #undef INIT1x8
            #undef SAVE1x8
#else
            int ldc_offset = ldc * sizeof(float) - 16;
                #define INIT1x8                         \
                "   mov r9,        %2               \n" \
                "   vld1.f32 {d16,d17}, [r9]!       \n" \
                "   vld1.f32 {d20,d21}, [r9]        \n" \
                "   vmov.u32 q9,   #0               \n" \
                "   vmov.u32 q11,  #0               \n" 

                #define SAVE1x8                         \
                "   mov r9,       %2                \n" \
                "   vadd.f32 q8,  q8,  q9           \n" \
                "   vadd.f32 q10, q10, q11          \n" \
                "   vst1.f32 {d16,d17}, [r9]!       \n" \
                "   vst1.f32 {d20,d21}, [r9]        \n"
                
            asm volatile(
                "   vld1.f32 {d0,d1}, [%0]!         \n"
                "   vld1.f32 {d4,d5}, [%1]!         \n"
                INIT1x8
                "mov r8,%4                          \n"
                "0:                                 \n"
                "   subs r9, r8,  #4                \n"
                "   blt 1f                          \n"
                "   vld1.f32 {d2,d3},  [%0]!        \n"
                "   vmla.f32 q8,  q0, d4[0]         \n"
                "   vld1.f32 {d6,d7},  [%0]!        \n"
                "   vmla.f32 q10, q1, d4[0]         \n"
                "   vld1.f32 {d8,d9},  [%0]!        \n"
                "   vmla.f32 q9,  q3, d4[1]         \n"
                "   vld1.f32 {d0,d1},  [%0]!        \n"
                "   vmla.f32 q11, q4, d4[1]         \n"
                "   subs r8, r8,  #4                \n"

                "   vld1.f32 {d2,d3},  [%0]!        \n"
                "   vmla.f32 q8,  q0, d5[0]         \n"
                "   vld1.f32 {d6,d7},  [%0]!        \n"
                "   vmla.f32 q10, q1, d5[0]         \n"
                "   vld1.f32 {d8,d9},  [%0]!        \n"
                "   vmla.f32 q9,  q3, d5[1]         \n"
                "   vld1.f32 {d0,d1},  [%0]!        \n"
                "   vmla.f32 q11, q4, d5[1]         \n"
                "   vld1.f32 {d4,d5},  [%1]!        \n"
                "   bgt 0b                          \n"
                "1:                                 \n"
                "   subs r8, r8,  #1                \n"
                "   vld1.f32 {d2,d3},  [%0]!        \n"
                "   blt 2f                          \n"
                "   vmla.f32 q8,  q0, d4[0]         \n"
                "   vmla.f32 q10, q1, d4[0]         \n"
                "   sub %1, %1,   #12               \n"
                "   vld1.f32 {d0,d1},  [%0]!        \n"
                "   vld1.f32 {d4,d5},  [%1]!        \n"
                "   bne 1b                          \n"
                "2:                                 \n"
                SAVE1x8
                "                                   \n"
                : "=r"(b), "=r"(a), "=r"(c), "=r"(ldc_offset), "=r"(k)
                : "0"(b), "1"(a), "2"(c), "3"(ldc_offset), "4"(k)
                : "memory", "cc", "r8", "r9", "q0", "q1", "q2", "q3", "q4", "q8", "q9", "q10", "q11");
#endif  // __aarch64__
        }
        int remain = n % 8;
        if (remain) {
            const float *a = ar;
            const float *b = br + (n / 8) * 8 * k;
            float *c       = cr + (n / 8) * 8;
            float32x4_t c0 = vdupq_n_f32(0);
            float32x4_t c1 = vdupq_n_f32(0);
            for (int kk = 0; kk < k; ++kk) {
                float32x4_t b0 = vld1q_f32(b);
                float32x4_t b1 = vld1q_f32(b + 4);
                float32x4_t a0 = vdupq_n_f32(a[kk]);
                c0             = vmlaq_f32(c0, a0, b0);
                c1             = vmlaq_f32(c1, a0, b1);
                b += 8;
            }
            Float4 c0_w(c0);
            Float4 c1_w(c1);
            for (int rr = 0; rr < remain; ++rr) {
                c[rr] += rr < 4 ? c0_w[rr] : c1_w[rr - 4];
            }
        }
    }
#else
    return NaiveKernel<1, 8>(m, n, k, sa, sb, sc, ldc);
#endif  // TNN_USE_NEON
}

template <int nr, typename T>
void NaivePackB(int k, int n, const T *from, int ldb, T *to) {
    const T *src = from;
    T *dst;

    for (int j = 0; j < k; ++j) {
        dst = to + j * nr;
        src = from + j * ldb;
        for (int i = 0; i < n / nr; i++) {
            for (int r = 0; r < nr; ++r) {
                dst[r] = src[r];
            }
            src += nr;
            dst += k * nr;
        }
        int remain = n % nr;
        if (remain) {
            for (int r = 0; r < remain; ++r) {
                dst[r] = src[r];
            }
            for (int r = remain; r < nr; ++r) {
                dst[r] = 0;
            }
        }
    }
}

#if TNN_ARM82
template void NaivePackB<16, fp16_t>(int k, int n, const fp16_t *from, int ldb, fp16_t *to);
#endif  // TNN_ARM82

void PackB_8(int k, int n, const float *from, int ldb, float *to) {
#ifdef TNN_USE_NEON
    int j = 0;

    const float *src[4];
    float *dst;
    float32x4_t val[8];
    for (int loop = 0; loop < k / 4; ++loop, j += 4) {
        dst    = to + j * 8;
        src[0] = from + j * ldb;
        src[1] = src[0] + ldb;
        src[2] = src[1] + ldb;
        src[3] = src[2] + ldb;
        for (int i = 0; i < n / 8; i++) {
            val[0] = vld1q_f32(src[0]);
            val[1] = vld1q_f32(src[0] + 4);
            src[0] += 8;

            val[2] = vld1q_f32(src[1]);
            val[3] = vld1q_f32(src[1] + 4);
            src[1] += 8;

            val[4] = vld1q_f32(src[2]);
            val[5] = vld1q_f32(src[2] + 4);
            src[2] += 8;

            val[6] = vld1q_f32(src[3]);
            val[7] = vld1q_f32(src[3] + 4);
            src[3] += 8;

            vst1q_f32(dst, val[0]);
            vst1q_f32(dst + 4, val[1]);
            vst1q_f32(dst + 8, val[2]);
            vst1q_f32(dst + 12, val[3]);
            vst1q_f32(dst + 16, val[4]);
            vst1q_f32(dst + 20, val[5]);
            vst1q_f32(dst + 24, val[6]);
            vst1q_f32(dst + 28, val[7]);
            dst += k * 8;
        }
        int remain = n % 8;
        if (remain) {
            for (int kr = 0; kr < 4; ++kr) {
                for (int r = 0; r < remain; ++r) {
                    dst[r] = src[kr][r];
                }
                for (int r = remain; r < 8; ++r) {
                    dst[r] = 0;
                }
                dst += 8;
            }
        }
    }

    for (; j < k; ++j) {
        dst    = to + j * 8;
        src[0] = from + j * ldb;
        for (int i = 0; i < n / 8; i++) {
            val[0] = vld1q_f32(src[0]);
            val[1] = vld1q_f32(src[0] + 4);
            src[0] += 8;

            vst1q_f32(dst, val[0]);
            vst1q_f32(dst + 4, val[1]);
            dst += k * 8;
        }
        int remain = n % 8;
        if (remain) {
            for (int r = 0; r < remain; ++r) {
                dst[r] = src[0][r];
            }
            for (int r = remain; r < 8; ++r) {
                dst[r] = 0;
            }
        }
    }
#else
    return NaivePackB<8>(k, n, from, ldb, to);
#endif
}

void PackA_12(int m, int k, const float *src, int lda, float *dst) {
    const float *src_offset[12];
    for (int j = 0; j < m - 11; j += 12) {
        float *dst_r   = dst + j * k;
        src_offset[0]  = src;
        src_offset[1]  = src_offset[0] + lda;
        src_offset[2]  = src_offset[1] + lda;
        src_offset[3]  = src_offset[2] + lda;
        src_offset[4]  = src_offset[3] + lda;
        src_offset[5]  = src_offset[4] + lda;
        src_offset[6]  = src_offset[5] + lda;
        src_offset[7]  = src_offset[6] + lda;
        src_offset[8]  = src_offset[7] + lda;
        src_offset[9]  = src_offset[8] + lda;
        src_offset[10] = src_offset[9] + lda;
        src_offset[11] = src_offset[10] + lda;
        src += 12 * lda;

        OMP_PARALLEL_FOR_
        for (int i = 0; i < k; ++i) {
            float *dst_t  = dst_r + i * 12;
            *(dst_t + 0)  = *(src_offset[0] + i);
            *(dst_t + 1)  = *(src_offset[1] + i);
            *(dst_t + 2)  = *(src_offset[2] + i);
            *(dst_t + 3)  = *(src_offset[3] + i);
            *(dst_t + 4)  = *(src_offset[4] + i);
            *(dst_t + 5)  = *(src_offset[5] + i);
            *(dst_t + 6)  = *(src_offset[6] + i);
            *(dst_t + 7)  = *(src_offset[7] + i);
            *(dst_t + 8)  = *(src_offset[8] + i);
            *(dst_t + 9)  = *(src_offset[9] + i);
            *(dst_t + 10) = *(src_offset[10] + i);
            *(dst_t + 11) = *(src_offset[11] + i);
        }
    }
}

void PackA_4(int m, int k, const float *src, int lda, float *dst) {
    const float *src_offset[4];
    for (int j = 0; j < m - 3; j += 4) {
        float *dst_r  = dst + j * k;
        src_offset[0] = src;
        src_offset[1] = src_offset[0] + lda;
        src_offset[2] = src_offset[1] + lda;
        src_offset[3] = src_offset[2] + lda;
        src += 4 * lda;

        OMP_PARALLEL_FOR_
        for (int i = 0; i < k; ++i) {
            float *dst_t = dst_r + i * 4;
            *(dst_t + 0) = *(src_offset[0] + i);
            *(dst_t + 1) = *(src_offset[1] + i);
            *(dst_t + 2) = *(src_offset[2] + i);
            *(dst_t + 3) = *(src_offset[3] + i);
        }
    }
}

void PackA_1(int m, int k, const float *src, int lda, float *dst) {
    OMP_PARALLEL_FOR_
    for (int j = 0; j < m; ++j) {
        memcpy(dst + j * k, src + j * lda, k * sizeof(float));
    }
}

}  // namespace TNN_NS
