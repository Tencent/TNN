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
    int do_relu = act_type == 1 || act_type == 2;

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
            PostAddBiasSwish<T, true>(dst, nullptr, plane_num, oc4);
        else
            PostAddBiasSwish<T, false>(dst, nullptr, plane_num, oc4);
    }
}

template void sgemm_repack_lhs(float *dst, float *src, float *weight, int ic4, int oc4, int plane_num, int dst_z_step,
                               int a_block, int b_block, float *work_space, float *bias, int act_type, bool fast_post);

template void sgemm_repack_lhs(bfp16_t *dst, bfp16_t *src, float *weight, int ic4, int oc4, int plane_num,
                               int dst_z_step, int a_block, int b_block, bfp16_t *work_space, float *bias,
                               int act_type, bool fast_post);

template <typename T>
void sgemm_repack_rhs(T *dst, T *src, float *weight, int ic4, int oc4, int plane_num, int dst_z_step, int a_block,
                      int b_block, T *work_space, float *bias, int act_type, bool fast_post) {
    int loop   = plane_num / a_block;
    int remain = plane_num % a_block;
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
            PostAddBiasSwish<T, true>(dst, nullptr, plane_num, oc4);
        else
            PostAddBiasSwish<T, false>(dst, nullptr, plane_num, oc4);
    }
}

template void sgemm_repack_rhs(float *dst, float *src, float *weight, int ic4, int oc4, int plane_num, int dst_z_step,
                               int a_block, int b_block, float *work_space, float *bias, int act_type, bool fast_post);

template void sgemm_repack_rhs(bfp16_t *dst, bfp16_t *src, float *weight, int ic4, int oc4, int plane_num,
                               int dst_z_step, int a_block, int b_block, bfp16_t *work_space, float *bias,
                               int act_type, bool fast_post);

}  // namespace TNN_NS
