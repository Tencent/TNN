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
#if TNN_ARM82

#include "tnn/device/arm/acc/Half8.h"
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

void Kernel_8x16(int m, int n, int k, const fp16_t *sa, const fp16_t *sb, fp16_t *sc, int ldc) {
#ifdef TNN_ARM82_A64
    for (int i = 0; i < m - 7; i += 8) {
        const fp16_t *ar = sa + i * k;
        const fp16_t *br = sb;
        fp16_t *cr       = sc + i * ldc;
        OMP_PARALLEL_FOR_
        for (int j = 0; j < n - 15; j += 16) {
            const fp16_t *a    = ar;
            const fp16_t *b    = br + j * k;
            fp16_t *c          = cr + j;
            int64_t ldc_offset = ldc * sizeof(fp16_t) - 16;
            int64_t k_64       = k;
            asm volatile(
                ".macro INIT8x16H                   \n"
                "   mov x9,        %2               \n"
                "   ld1 {v8.8h},  [x9], #16         \n"
                "   ld1 {v20.8h}, [x9], %3          \n"
                "   ld1 {v9.8h},  [x9], #16         \n"
                "   ld1 {v21.8h}, [x9], %3          \n"
                "   ld1 {v10.8h}, [x9], #16         \n"
                "   ld1 {v22.8h}, [x9], %3          \n"
                "   ld1 {v11.8h}, [x9], #16         \n"
                "   ld1 {v23.8h}, [x9], %3          \n"
                "   ld1 {v12.8h}, [x9], #16         \n"
                "   ld1 {v24.8h}, [x9], %3          \n"
                "   ld1 {v13.8h}, [x9], #16         \n"
                "   ld1 {v25.8h}, [x9], %3          \n"
                "   ld1 {v14.8h}, [x9], #16         \n"
                "   ld1 {v26.8h}, [x9], %3          \n"
                "   ld1 {v15.8h}, [x9], #16         \n"
                "   ld1 {v27.8h}, [x9]              \n"
                ".endm                              \n"
                "                                   \n"
                ".macro SAVE8x16H                   \n"
                "   mov x9,        %2               \n"
                "   st1 {v8.8h},  [x9], #16         \n"
                "   st1 {v20.8h}, [x9], %3          \n"
                "   st1 {v9.8h},  [x9], #16         \n"
                "   st1 {v21.8h}, [x9], %3          \n"
                "   st1 {v10.8h}, [x9], #16         \n"
                "   st1 {v22.8h}, [x9], %3          \n"
                "   st1 {v11.8h}, [x9], #16         \n"
                "   st1 {v23.8h}, [x9], %3          \n"
                "   st1 {v12.8h}, [x9], #16         \n"
                "   st1 {v24.8h}, [x9], %3          \n"
                "   st1 {v13.8h}, [x9], #16         \n"
                "   st1 {v25.8h}, [x9], %3          \n"
                "   st1 {v14.8h}, [x9], #16         \n"
                "   st1 {v26.8h}, [x9], %3          \n"
                "   st1 {v15.8h}, [x9], #16         \n"
                "   st1 {v27.8h}, [x9]              \n"
                ".endm                              \n"
                "                                   \n"
                "   ld1 {v0.8h}, [%0], #16          \n"
                "   ld1 {v2.8h}, [%1], #16          \n"
                "INIT8x16H                          \n"
                "mov x8,%4                          \n"
                "0:                                 \n"

                "   fmla v8.8h , v0.8h, v2.h[0]     \n"
                "   fmla v9.8h , v0.8h, v2.h[1]     \n"
                "   fmla v10.8h, v0.8h, v2.h[2]     \n"
                "   fmla v11.8h, v0.8h, v2.h[3]     \n"

                "   fmla v12.8h, v0.8h, v2.h[4]     \n"
                "   ld1 {v1.8h}, [%0], #16          \n"
                "   fmla v13.8h, v0.8h, v2.h[5]     \n"
                "   fmla v14.8h, v0.8h, v2.h[6]     \n"
                "   prfm pldl1keep, [%1, #64]       \n"
                "   fmla v15.8h, v0.8h, v2.h[7]     \n"

                "   fmla v20.8h, v1.8h, v2.h[0]     \n"
                "   ld1 {v0.8h}, [%0], #16          \n"
                "   fmla v21.8h, v1.8h, v2.h[1]     \n"
                "   fmla v22.8h, v1.8h, v2.h[2]     \n"
                "   prfm pldl1keep, [%0, #128]      \n"
                "   fmla v23.8h, v1.8h, v2.h[3]     \n"
                "   subs x8, x8, #1                 \n"

                "   fmla v24.8h, v1.8h, v2.h[4]     \n"
                "   fmla v25.8h, v1.8h, v2.h[5]     \n"
                "   fmla v26.8h, v1.8h, v2.h[6]     \n"
                "   fmla v27.8h, v1.8h, v2.h[7]     \n"

                "   ld1 {v2.8h}, [%1], #16          \n"
                "   bne 0b                          \n"
                "SAVE8x16H                          \n"
                "                                   \n"
                : "=r"(b), "=r"(a), "=r"(c), "=r"(ldc_offset), "=r"(k_64)
                : "0"(b), "1"(a), "2"(c), "3"(ldc_offset), "4"(k_64)
                : "memory", "cc", "x8", "x9", "v0", "v1", "v2", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                  "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
        }
        int remain = n % 16;
        if (remain) {
            const fp16_t *a   = ar;
            const fp16_t *b   = br + (n / 16) * 16 * k;
            fp16_t *c         = cr + (n / 16) * 16;
            float16x8_t c0[8] = {vdupq_n_f16(0)};
            float16x8_t c1[8] = {vdupq_n_f16(0)};
            float16x8_t b0, b1, av, a0, a1, a2, a3, a4, a5, a6, a7;
            for (int kk = 0; kk < k; ++kk) {
                b0    = vld1q_f16(b);
                b1    = vld1q_f16(b + 8);
                av    = vld1q_f16(a);
                a0    = vdupq_n_f16(av[0]);
                a1    = vdupq_n_f16(av[1]);
                a2    = vdupq_n_f16(av[2]);
                a3    = vdupq_n_f16(av[3]);
                a4    = vdupq_n_f16(av[4]);
                a5    = vdupq_n_f16(av[5]);
                a6    = vdupq_n_f16(av[6]);
                a7    = vdupq_n_f16(av[7]);
                c0[0] = vfmaq_f16(c0[0], a0, b0);
                c1[0] = vfmaq_f16(c1[0], a0, b1);
                c0[1] = vfmaq_f16(c0[1], a1, b0);
                c1[1] = vfmaq_f16(c1[1], a1, b1);
                c0[2] = vfmaq_f16(c0[2], a2, b0);
                c1[2] = vfmaq_f16(c1[2], a2, b1);
                c0[3] = vfmaq_f16(c0[3], a3, b0);
                c1[3] = vfmaq_f16(c1[3], a3, b1);
                c0[4] = vfmaq_f16(c0[4], a4, b0);
                c1[4] = vfmaq_f16(c1[4], a4, b1);
                c0[5] = vfmaq_f16(c0[5], a5, b0);
                c1[5] = vfmaq_f16(c1[5], a5, b1);
                c0[6] = vfmaq_f16(c0[6], a6, b0);
                c1[6] = vfmaq_f16(c1[6], a6, b1);
                c0[7] = vfmaq_f16(c0[7], a7, b0);
                c1[7] = vfmaq_f16(c1[7], a7, b1);

                b += 16;
                a += 8;
            }
            for (int ms = 0; ms < 8; ++ms) {
                for (int rr = 0; rr < remain; ++rr) {
                    c[rr] += rr < 8 ? c0[ms][rr] : c1[ms][rr - 8];
                }
                c += ldc;
            }
        }
    }
#else
    return NaiveKernel<8, 16>(m, n, k, sa, sb, sc, ldc);
#endif  // TNN_ARM82_A64
}

void Kernel_4x16(int m, int n, int k, const fp16_t *sa, const fp16_t *sb, fp16_t *sc, int ldc) {
#ifdef TNN_ARM82_USE_NEON
    for (int i = 0; i < m - 3; i += 4) {
        const fp16_t *ar = sa + i * k;
        const fp16_t *br = sb;
        fp16_t *cr       = sc + i * ldc;
        OMP_PARALLEL_FOR_
        for (int j = 0; j < n - 15; j += 16) {
            const fp16_t *a = ar;
            const fp16_t *b = br + j * k;
            fp16_t *c       = cr + j;
#ifdef TNN_ARM82_A64
            int64_t ldc_offset = ldc * sizeof(fp16_t) - 16;
            int64_t k_64       = k;
            asm volatile(
                ".macro INIT4x16H                   \n"
                "   mov x9,        %2               \n"
                "   ld1 {v8.8h},  [x9], #16         \n"
                "   ld1 {v20.8h}, [x9], %3          \n"
                "   ld1 {v9.8h},  [x9], #16         \n"
                "   ld1 {v21.8h}, [x9], %3          \n"
                "   ld1 {v10.8h}, [x9], #16         \n"
                "   ld1 {v22.8h}, [x9], %3          \n"
                "   ld1 {v11.8h}, [x9], #16         \n"
                "   ld1 {v23.8h}, [x9]              \n"
                ".endm                              \n"
                "                                   \n"
                ".macro SAVE4x16H                   \n"
                "   mov x9,        %2               \n"
                "   st1 {v8.8h},  [x9], #16         \n"
                "   st1 {v20.8h}, [x9], %3          \n"
                "   st1 {v9.8h},  [x9], #16         \n"
                "   st1 {v21.8h}, [x9], %3          \n"
                "   st1 {v10.8h}, [x9], #16         \n"
                "   st1 {v22.8h}, [x9], %3          \n"
                "   st1 {v11.8h}, [x9], #16         \n"
                "   st1 {v23.8h}, [x9]              \n"
                ".endm                              \n"
                "                                   \n"
                "   ld1 {v0.8h}, [%0], #16          \n"
                "   ld1 {v2.4h}, [%1], #8           \n"
                "INIT4x16H                          \n"
                "mov x8,%4                          \n"
                "0:                                 \n"

                "   fmla v8.8h , v0.8h, v2.h[0]     \n"
                "   fmla v9.8h , v0.8h, v2.h[1]     \n"
                "   fmla v10.8h, v0.8h, v2.h[2]     \n"
                "   fmla v11.8h, v0.8h, v2.h[3]     \n"

                "   ld1 {v1.8h}, [%0], #16          \n"
                "   prfm pldl1keep, [%1, #64]       \n"
                "   prfm pldl1keep, [%0, #64]       \n"

                "   fmla v20.8h, v1.8h, v2.h[0]     \n"
                "   ld1 {v0.8h}, [%0], #16          \n"
                "   fmla v21.8h, v1.8h, v2.h[1]     \n"
                "   fmla v22.8h, v1.8h, v2.h[2]     \n"
                "   prfm pldl1keep, [%0, #128]      \n"
                "   fmla v23.8h, v1.8h, v2.h[3]     \n"
                "   subs x8, x8, #1                 \n"
                "   ld1 {v2.4h}, [%1], #8           \n"
                "   bne 0b                          \n"
                "SAVE4x16H                          \n"
                "                                   \n"
                : "=r"(b), "=r"(a), "=r"(c), "=r"(ldc_offset), "=r"(k_64)
                : "0"(b), "1"(a), "2"(c), "3"(ldc_offset), "4"(k_64)
                : "memory", "cc", "x8", "x9", "v0", "v1", "v2", "v8", "v9", "v10", "v11", "v20", "v21", "v22", "v23");
#else
            int ldc_offset = ldc * sizeof(fp16_t) - 16;
            asm volatile(
                ".macro INIT4x16H                   \n"
                "   mov r9,        %2               \n"
                "   vld1.16 {d16,d17},  [r9]!       \n"
                "   vld1.16 {d18,d19},  [r9]        \n"
                "   add      r9,   r9, %3           \n"
                "   vld1.16 {d20,d21}, [r9]!        \n"
                "   vld1.16 {d22,d23}, [r9]         \n"
                "   add      r9,   r9, %3           \n"
                "   vld1.16 {d24,d25}, [r9]!        \n"
                "   vld1.16 {d26,d27}, [r9]         \n"
                "   add      r9,   r9, %3           \n"
                "   vld1.16 {d28,d29}, [r9]!        \n"
                "   vld1.16 {d30,d31}, [r9]         \n"
                ".endm                              \n"
                "                                   \n"
                ".macro SAVE4x16H                   \n"
                "   mov r9,        %2               \n"
                "   vst1.16 {d16,d17},  [r9]!       \n"
                "   vst1.16 {d18,d19},  [r9]        \n"
                "   add      r9,   r9, %3           \n"
                "   vst1.16 {d20,d21}, [r9]!        \n"
                "   vst1.16 {d22,d23}, [r9]         \n"
                "   add      r9,   r9, %3           \n"
                "   vst1.16 {d24,d25}, [r9]!        \n"
                "   vst1.16 {d26,d27}, [r9]         \n"
                "   add      r9,   r9, %3           \n"
                "   vst1.16 {d28,d29}, [r9]!        \n"
                "   vst1.16 {d30,d31}, [r9]         \n"
                ".endm                              \n"
                "                                   \n"
                "   vld1.16 {d0,d1},  [%0]!         \n"
                "   vld1.16 {d4},     [%1]!         \n"
                "INIT4x16H                          \n"
                "mov r8,%4                          \n"
                "0:                                 \n"
                "   vmla.f16 q8,  q0, d4[0]         \n"
                "   vld1.16 {d2,d3},  [%0]!         \n"
                "   vmla.f16 q10, q0, d4[1]         \n"
                "   vmla.f16 q12, q0, d4[2]         \n"
                "   vmla.f16 q14, q0, d4[3]         \n"
                "   subs r8, r8, #1                 \n"

                "   vmla.f16 q9,  q1, d4[0]         \n"
                "   vmla.f16 q11, q1, d4[1]         \n"
                "   vmla.f16 q13, q1, d4[2]         \n"
                "   vld1.16 {d0,d1},  [%0]!         \n"
                "   vmla.f16 q15, q1, d4[3]         \n"
                "   vld1.16 {d4},     [%1]!         \n"
                "   bne 0b                          \n"
                "SAVE4x16H                          \n"
                "                                   \n"
                : "=r"(b), "=r"(a), "=r"(c), "=r"(ldc_offset), "=r"(k)
                : "0"(b), "1"(a), "2"(c), "3"(ldc_offset), "4"(k)
                : "memory", "cc", "r8", "r9", "q0", "q1", "q2", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif  // TNN_ARM82_A64
        }
        int remain = n % 16;
        if (remain) {
            const fp16_t *a = ar;
            const fp16_t *b = br + (n / 16) * 16 * k;
            fp16_t *c       = cr + (n / 16) * 16;
            Half8 vec_0     = Half8(fp16_t(0));
            Half8 c0[4]     = {vec_0, vec_0, vec_0, vec_0};
            Half8 c1[4]     = {vec_0, vec_0, vec_0, vec_0};
            Half8 b0, b1, av, a0, a1, a2, a3;
            for (int kk = 0; kk < k; ++kk) {
                b0 = Half8::load(b);
                b1 = Half8::load(b + 8);
                av = Half8::load(a);
                a0 = Half8(av[0]);
                a1 = Half8(av[1]);
                a2 = Half8(av[2]);
                a3 = Half8(av[3]);
                Half8::mla(c0[0], a0, b0);
                Half8::mla(c1[0], a0, b1);
                Half8::mla(c0[1], a1, b0);
                Half8::mla(c1[1], a1, b1);
                Half8::mla(c0[2], a2, b0);
                Half8::mla(c1[2], a2, b1);
                Half8::mla(c0[3], a3, b0);
                Half8::mla(c1[3], a3, b1);
                b += 16;
                a += 4;
            }
            for (int ms = 0; ms < 4; ++ms) {
                if (remain > 8) {
                    Half8::save(c, c0[ms] + Half8::load(c));
                    Half8 c_old = vec_0;
                    for (int rr = 8; rr < remain; ++rr) {
                        c_old.set_lane(c[rr], rr - 8);
                    }
                    Half8 c_new = c_old + c1[ms];
                    for (int rr = 8; rr < remain; ++rr) {
                        c[rr] = c_new[rr - 8];
                    }
                } else {
                    Half8 c_old = vec_0;
                    for (int rr = 0; rr < remain; ++rr) {
                        c_old.set_lane(c[rr], rr);
                    }
                    Half8 c_new = c_old + c0[ms];
                    for (int rr = 0; rr < remain; ++rr) {
                        c[rr] = c_new[rr];
                    }
                }
                c += ldc;
            }
        }
    }
#else
    return NaiveKernel<4, 16>(m, n, k, sa, sb, sc, ldc);
#endif  // TNN_ARM82_USE_NEON
}

void Kernel_1x16(int m, int n, int k, const fp16_t *sa, const fp16_t *sb, fp16_t *sc, int ldc) {
#ifdef TNN_ARM82_USE_NEON
    for (int i = 0; i < m; ++i) {
        const fp16_t *ar = sa + i * k;
        const fp16_t *br = sb;
        fp16_t *cr       = sc + i * ldc;
        OMP_PARALLEL_FOR_
        for (int j = 0; j < n - 15; j += 16) {
            const fp16_t *a = ar;
            const fp16_t *b = br + j * k;
            fp16_t *c       = cr + j;
#ifdef TNN_ARM82_A64
            int64_t ldc_offset = ldc * sizeof(fp16_t) - 16;
            int64_t k_64       = k;
            asm volatile(
                ".macro INIT1x16H                   \n"
                "   mov x9,        %2               \n"
                "   ld1 {v8.8h},  [x9], #16         \n"
                "   ld1 {v20.8h}, [x9], %3          \n"
                "   movi v9.8h,    #0               \n"
                "   movi v21.8h,   #0               \n"
                ".endm                              \n"
                "                                   \n"
                ".macro SAVE1x16H                   \n"
                "   mov x9,        %2               \n"
                "   fadd v8.8h,  v8.8h,  v9.8h      \n"
                "   fadd v20.8h, v20.8h, v21.8h     \n"
                "   st1 {v8.8h},  [x9], #16         \n"
                "   st1 {v20.8h}, [x9], %3          \n"
                ".endm                              \n"
                "                                   \n"
                "   ld1 {v0.8h}, [%0], #16          \n"
                "   ld1 {v2.4h}, [%1], #8           \n"
                "INIT1x16H                          \n"
                "mov x8,%4                          \n"
                "0:                                 \n"
                "   subs x9, x8, #4                 \n"
                "   blt 1f                          \n"
                "   ld1 {v1.8h}, [%0], #16          \n"
                "   fmla v8.8h , v0.8h, v2.h[0]     \n"
                "   ld1 {v3.8h}, [%0], #16          \n"
                "   fmla v20.8h, v1.8h, v2.h[0]     \n"
                "   ld1 {v4.8h}, [%0], #16          \n"
                "   fmla v9.8h , v3.8h, v2.h[1]     \n"
                "   ld1 {v0.8h}, [%0], #16          \n"
                "   fmla v21.8h, v4.8h, v2.h[1]     \n"
                "   subs x8, x8, #4                 \n"

                "   ld1 {v1.8h}, [%0], #16          \n"
                "   fmla v8.8h , v0.8h, v2.h[2]     \n"
                "   ld1 {v3.8h}, [%0], #16          \n"
                "   fmla v20.8h, v1.8h, v2.h[2]     \n"
                "   ld1 {v4.8h}, [%0], #16          \n"
                "   fmla v9.8h , v3.8h, v2.h[3]     \n"
                "   ld1 {v0.8h}, [%0], #16          \n"
                "   fmla v21.8h, v4.8h, v2.h[3]     \n"
                "   ld1 {v2.4h}, [%1], #8           \n"
                "   bgt 0b                          \n"
                "1:                                 \n"
                "   subs x8, x8, #1                 \n"
                "   ld1 {v1.8h}, [%0], #16          \n"
                "   blt 2f                          \n"
                "   fmla v8.8h , v0.8h, v2.h[0]     \n"
                "   fmla v20.8h, v1.8h, v2.h[0]     \n"
                "   sub %1, %1, #6                  \n"
                "   ld1 {v0.8h}, [%0], #16          \n"
                "   ld1 {v2.4h}, [%1], #8           \n"
                "   bne 1b                          \n"
                "2:                                 \n"
                "   SAVE1x16H                       \n"
                "                                   \n"
                : "=r"(b), "=r"(a), "=r"(c), "=r"(ldc_offset), "=r"(k_64)
                : "0"(b), "1"(a), "2"(c), "3"(ldc_offset), "4"(k_64)
                : "memory", "cc", "x8", "x9", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v20", "v21");
#else
            int ldc_offset = ldc * sizeof(float) - 16;
            asm volatile(
                ".macro INIT1x16H                   \n"
                "   mov r9,        %2               \n"
                "   vld1.16 {d16,d17}, [r9]!        \n"
                "   vld1.16 {d20,d21}, [r9]         \n"
                "   vmov.u32 q9,   #0               \n"
                "   vmov.u32 q11,  #0               \n"
                ".endm                              \n"
                "                                   \n"
                ".macro SAVE1x16H                   \n"
                "   mov r9,       %2                \n"
                "   vadd.f16 q8,  q8,  q9           \n"
                "   vadd.f16 q10, q10, q11          \n"
                "   vst1.16 {d16,d17}, [r9]!        \n"
                "   vst1.16 {d20,d21}, [r9]         \n"
                ".endm                              \n"
                "                                   \n"
                "   vld1.16 {d0,d1}, [%0]!          \n"
                "   vld1.16 {d4},    [%1]!          \n"
                "INIT1x16H                          \n"
                "mov r8,%4                          \n"
                "0:                                 \n"
                "   subs r9, r8,  #4                \n"
                "   blt 1f                          \n"
                "   vld1.16 {d2,d3},  [%0]!         \n"
                "   vmla.f16 q8,  q0, d4[0]         \n"
                "   vld1.16 {d6,d7},  [%0]!         \n"
                "   vmla.f16 q10, q1, d4[0]         \n"
                "   vld1.16 {d8,d9},  [%0]!         \n"
                "   vmla.f16 q9,  q3, d4[1]         \n"
                "   vld1.16 {d0,d1},  [%0]!         \n"
                "   vmla.f16 q11, q4, d4[1]         \n"
                "   subs r8, r8,  #4                \n"

                "   vld1.16 {d2,d3},  [%0]!         \n"
                "   vmla.f16 q8,  q0, d4[2]         \n"
                "   vld1.16 {d6,d7},  [%0]!         \n"
                "   vmla.f16 q10, q1, d4[2]         \n"
                "   vld1.16 {d8,d9},  [%0]!         \n"
                "   vmla.f16 q9,  q3, d4[3]         \n"
                "   vld1.16 {d0,d1},  [%0]!         \n"
                "   vmla.f16 q11, q4, d4[3]         \n"
                "   vld1.16 {d4},     [%1]!         \n"
                "   bgt 0b                          \n"
                "1:                                 \n"
                "   subs r8, r8,  #1                \n"
                "   vld1.16 {d2,d3},  [%0]!         \n"
                "   blt 2f                          \n"
                "   vmla.f16 q8,  q0, d4[0]         \n"
                "   vmla.f16 q10, q1, d4[0]         \n"
                "   sub %1, %1,   #6                \n"
                "   vld1.16 {d0,d1},  [%0]!         \n"
                "   vld1.16 {d4},     [%1]!         \n"
                "   bne 1b                          \n"
                "2:                                 \n"
                "   SAVE1x16H                       \n"
                "                                   \n"
                : "=r"(b), "=r"(a), "=r"(c), "=r"(ldc_offset), "=r"(k)
                : "0"(b), "1"(a), "2"(c), "3"(ldc_offset), "4"(k)
                : "memory", "cc", "r8", "r9", "q0", "q1", "q2", "q3", "q4", "q8", "q9", "q10", "q11");
#endif  // TNN_ARM82_A64
        }
        int remain = n % 16;
        if (remain) {
            const fp16_t *a = ar;
            const fp16_t *b = br + (n / 16) * 16 * k;
            fp16_t *c       = cr + (n / 16) * 16;
            Half8 vec_0     = Half8(fp16_t(0));
            Half8 c0        = vec_0;
            Half8 c1        = vec_0;
            // Kahan summation
            // set to zero
            Half8 error_0 = vec_0;
            Half8 error_1 = vec_0;
            for (int kk = 0; kk < k; ++kk) {
                Half8 b0 = Half8::load(b);
                Half8 b1 = Half8::load(b + 8);
                Half8 a0 = Half8(a[kk]);
                Half8 y_0 = a0 * b0 - error_0;
                Half8 y_1 = a0 * b1 - error_1;
                Half8 t_0 = c0 + y_0;
                Half8 t_1 = c1 + y_1;
                error_0 = (t_0 - c0) - y_0;
                error_1 = (t_1 - c1) - y_1;
                c0 = t_0;
                c1 = t_1;
                b += 16;
            }
            if (remain > 8) {
                Half8::save(c, c0 + Half8::load(c));
                Half8 c_old = vec_0;
                for (int rr = 8; rr < remain; ++rr) {
                    c_old.set_lane(c[rr], rr - 8);
                }
                Half8 c_new = c_old + c1;
                for (int rr = 8; rr < remain; ++rr) {
                    c[rr] = c_new[rr - 8];
                }
            } else {
                Half8 c_old = vec_0;
                for (int rr = 0; rr < remain; ++rr) {
                    c_old.set_lane(c[rr], rr);
                }
                Half8 c_new = c_old + c0;
                for (int rr = 0; rr < remain; ++rr) {
                    c[rr] = c_new[rr];
                }
            }
        }
    }
#else
    return NaiveKernel<1, 16>(m, n, k, sa, sb, sc, ldc);
#endif  // TNN_ARM82_USE_NEON
}

void PackB_16(int k, int n, const fp16_t *from, int ldb, fp16_t *to) {
    return NaivePackB<16>(k, n, from, ldb, to);
}

void PackA_8(int m, int k, const fp16_t *src, int lda, fp16_t *dst) {
    const fp16_t *src_offset[8];
    for (int j = 0; j < m - 7; j += 8) {
        fp16_t *dst_r = dst + j * k;
        src_offset[0] = src;
        src_offset[1] = src_offset[0] + lda;
        src_offset[2] = src_offset[1] + lda;
        src_offset[3] = src_offset[2] + lda;
        src_offset[4] = src_offset[3] + lda;
        src_offset[5] = src_offset[4] + lda;
        src_offset[6] = src_offset[5] + lda;
        src_offset[7] = src_offset[6] + lda;
        src += 8 * lda;

        OMP_PARALLEL_FOR_
        for (int i = 0; i < k; ++i) {
            fp16_t *dst_t = dst_r + i * 8;
            *(dst_t + 0)  = *(src_offset[0] + i);
            *(dst_t + 1)  = *(src_offset[1] + i);
            *(dst_t + 2)  = *(src_offset[2] + i);
            *(dst_t + 3)  = *(src_offset[3] + i);
            *(dst_t + 4)  = *(src_offset[4] + i);
            *(dst_t + 5)  = *(src_offset[5] + i);
            *(dst_t + 6)  = *(src_offset[6] + i);
            *(dst_t + 7)  = *(src_offset[7] + i);
        }
    }
}

void PackA_4(int m, int k, const fp16_t *src, int lda, fp16_t *dst) {
    const fp16_t *src_offset[4];
    for (int j = 0; j < m - 3; j += 4) {
        fp16_t *dst_r = dst + j * k;
        src_offset[0] = src;
        src_offset[1] = src_offset[0] + lda;
        src_offset[2] = src_offset[1] + lda;
        src_offset[3] = src_offset[2] + lda;
        src += 4 * lda;

        OMP_PARALLEL_FOR_
        for (int i = 0; i < k; ++i) {
            fp16_t *dst_t = dst_r + i * 4;
            *(dst_t + 0)  = *(src_offset[0] + i);
            *(dst_t + 1)  = *(src_offset[1] + i);
            *(dst_t + 2)  = *(src_offset[2] + i);
            *(dst_t + 3)  = *(src_offset[3] + i);
        }
    }
}

void PackA_1(int m, int k, const fp16_t *src, int lda, fp16_t *dst) {
    OMP_PARALLEL_FOR_
    for (int j = 0; j < m; ++j) {
        memcpy(dst + j * k, src + j * lda, k * sizeof(fp16_t));
    }
}

}  // namespace TNN_NS

#endif  // TNN_ARM82
