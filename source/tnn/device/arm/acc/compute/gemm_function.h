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

#ifndef TNN_ARM_GEMM_FUNCTION_H_
#define TNN_ARM_GEMM_FUNCTION_H_
#include "tnn/core/macro.h"
#include "tnn/device/arm/acc/compute/compute.h"

namespace TNN_NS {

#ifdef __aarch64__
#define ARM_SGEMM_TILE_M 12
#define ARM_SGEMM_TILE_N 8
#else
#define ARM_SGEMM_TILE_M 8
#define ARM_SGEMM_TILE_N 4
#endif

template <typename T>
void GEMM_FUNC(T *dst, const T *src, const float *weight, int src_depth_quad, int dst_step, int dst_depth_quad,
               int width, float *bias, long relu);

void set_block_size(int &a_block, int &b_block, int l2_size, const int plane_num, const int oc_r4, const int ic_r4,
                    int byte_size);
template <typename T>
void sgemm_repack_lhs(T *dst, T *src, float *weight, int ic4, int oc4, int width, int dst_z_step, int a_block,
                      int b_block, T *work_space, float *bias, int act_type, bool fast_post);
template <typename T>
void sgemm_repack_rhs(T *dst, T *src, float *weight, int ic4, int oc4, int width, int dst_z_step, int a_block,
                      int b_block, T *work_space, float *bias, int act_type, bool fast_post);

void Kernel_12x8(int m, int n, int k, const float *sa, const float *sb, float *sc, int ldc);
void Kernel_4x8(int m, int n, int k, const float *sa, const float *sb, float *sc, int ldc);
void Kernel_1x8(int m, int n, int k, const float *sa, const float *sb, float *sc, int ldc);

void PackB_8(int k, int n, const float *src, int ldb, float *dst);
void PackA_12(int m, int k, const float *src, int lda, float *dst);
void PackA_4(int m, int k, const float *src, int lda, float *dst);
void PackA_1(int m, int k, const float *src, int lda, float *dst);

#if TNN_ARM82
template <int mr, int nr, typename T>
void NaiveKernel(int m, int n, int k, const T *sa, const T *sb, T *sc, int ldc);
void Kernel_8x16(int m, int n, int k, const fp16_t* sa, const fp16_t* sb, fp16_t* sc, int ldc);
void Kernel_4x16(int m, int n, int k, const fp16_t* sa, const fp16_t* sb, fp16_t* sc, int ldc);
void Kernel_1x16(int m, int n, int k, const fp16_t* sa, const fp16_t* sb, fp16_t* sc, int ldc);

template <int nr, typename T>
void NaivePackB(int k, int n, const T *from, int ldb, T *to);
void PackB_16(int k, int n, const fp16_t* src, int ldb, fp16_t* dst);
void PackA_8(int m, int k, const fp16_t* src, int lda, fp16_t* dst);
void PackA_4(int m, int k, const fp16_t* src, int lda, fp16_t* dst);
void PackA_1(int m, int k, const fp16_t* src, int lda, fp16_t* dst);
#endif

}  // namespace TNN_NS

#endif
