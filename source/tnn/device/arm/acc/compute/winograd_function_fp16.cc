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

#include "tnn/device/arm/acc/compute/winograd_function.h"
#include <cstring>
#include <memory>
#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

namespace TNN_NS {

static inline void WeightTransform(const __fp16 *src, __fp16 *dst, int kernel_size, int unit, int in_channel,
                                   int out_channel, const __fp16 (*G)[3]) {
    __fp16 M[unit][3];
    __fp16 K_trans[unit * unit];

    int ic_8        = UP_DIV(in_channel, 8);
    int oc_8        = UP_DIV(out_channel, 8);
    int unit_stride = ic_8 * oc_8 * 8 * 8;
    int oc_stride   = ic_8 * 8 * 8;
    int ic_stride   = 8 * 8;

    for (int oc = 0; oc < out_channel; oc++) {
        int zo        = oc / 8;
        int ro        = oc % 8;
        __fp16 *dst_oz = dst + zo * oc_stride + ro;
        for (int ic = 0; ic < in_channel; ic++) {
            const __fp16 *src_z = src + (oc * in_channel + ic) * 3 * 3;
            const __fp16 *k0    = src_z;
            const __fp16 *k1    = k0 + 3;
            const __fp16 *k2    = k1 + 3;

            int zi = ic / 8;
            int ri = ic % 8;

            // M=G*g
            for (int i = 0; i < unit; i++) {
                M[i][0] = k0[0] * G[i][0] + k1[0] * G[i][1] + k2[0] * G[i][2];
                M[i][1] = k0[1] * G[i][0] + k1[1] * G[i][1] + k2[1] * G[i][2];
                M[i][2] = k0[2] * G[i][0] + k1[2] * G[i][1] + k2[2] * G[i][2];
            }

            // K_trans=M*GT
            for (int j = 0; j < unit; j++) {
                __fp16 *Mp = &M[j][0];
                for (int i = 0; i < unit; i++) {
                    K_trans[j * unit + i] = Mp[0] * G[i][0] + Mp[1] * G[i][1] + Mp[2] * G[i][2];
                }
            }

            auto dst_sz = dst_oz + zi * ic_stride + 8 * ri;

            for (int k = 0; k < unit * unit; k++) {
                *(dst_sz + k * unit_stride) = K_trans[k];
            }
        }
    }
}

void WeightTransform4x4(const __fp16 *src, __fp16 *dst, int kernel_size, int in_channel, int out_channel) {
    const __fp16 G[4][3] = {{1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}};

    WeightTransform(src, dst, kernel_size, 4, in_channel, out_channel, G);
}

void WeightTransform6x6(const __fp16 *src, __fp16 *dst, int kernel_size, int in_channel, int out_channel) {
    const __fp16 G[6][3] = {
        {1.0f / 4, 0.0f, 0.0f},           {-1.0f / 6, -1.0f / 6, -1.0f / 6}, {-1.0f / 6, 1.0f / 6, -1.0f / 6},
        {1.0f / 24, 1.0f / 12, 1.0f / 6}, {1.0f / 24, -1.0f / 12, 1.0f / 6}, {0.0f, 0.0f, 1.0f}};

    WeightTransform(src, dst, kernel_size, 6, in_channel, out_channel, G);
}

void SrcTransformInOne4x4Fp16(const void *src, void *dst, int w_stride, int h_stride) {
    const __fp16 *src_ptr = reinterpret_cast<const __fp16 *>(src);
    __fp16 *dst_ptr = reinterpret_cast<__fp16 *>(dst);

    float16x8_t vec_src[4][4];
    float16x8_t vec_mid[4][4];
    float16x8_t vec_dst[4][4];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            vec_src[i][j] = vld1q_f16(src_ptr + i * h_stride + j * w_stride);
        }
    }

    for (int i = 0; i < 4; i++) {
        vec_mid[0][i] = vsubq_f16(vec_src[i][0], vec_src[i][2]);
        vec_mid[1][i] = vaddq_f16(vec_src[i][1], vec_src[i][2]);
        vec_mid[2][i] = vsubq_f16(vec_src[i][2], vec_src[i][1]);
        vec_mid[3][i] = vsubq_f16(vec_src[i][1], vec_src[i][3]);
    }

    for (int i = 0; i < 4; i++) {
        vec_dst[0][i] = vsubq_f16(vec_mid[i][0], vec_mid[i][2]);
        vec_dst[1][i] = vaddq_f16(vec_mid[i][1], vec_mid[i][2]);
        vec_dst[2][i] = vsubq_f16(vec_mid[i][2], vec_mid[i][1]);
        vec_dst[3][i] = vsubq_f16(vec_mid[i][1], vec_mid[i][3]);

        vst1q_f16(dst_ptr + 0  + 8 * i, vec_dst[0][i]);
        vst1q_f16(dst_ptr + 32 + 8 * i, vec_dst[1][i]);
        vst1q_f16(dst_ptr + 64 + 8 * i, vec_dst[2][i]);
        vst1q_f16(dst_ptr + 96 + 8 * i, vec_dst[3][i]);
    }
}

// A = [1  0]
//     [1  1]
//     [1 -1]
//     [0 -1]
void DstTransformInOne4x2Fp16(const void *src, void *dst, int w_stride, int h_stride, int ey) {
    const __fp16 *src_ptr = reinterpret_cast<const __fp16 *>(src);
    __fp16 *dst_ptr = reinterpret_cast<__fp16 *>(dst);

    float16x8_t vec_src[4][4];
    float16x8_t vec_mid[4][2];
    float16x8_t vec_dst[2][2];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            vec_src[i][j] = vld1q_f16(src_ptr + i * 4 * w_stride + j * w_stride);
        }
    }

    for (int i = 0; i < 4; i++) {
        vec_mid[i][0] = vaddq_f16(vaddq_f16(vec_src[0][i], vec_src[1][i]), vec_src[2][i]);
        vec_mid[i][1] = vsubq_f16(vsubq_f16(vec_src[1][i], vec_src[2][i]), vec_src[3][i]);
    }

    for (int i = 0; i < ey; i++) {
        vec_dst[i][0] = vaddq_f16(vaddq_f16(vec_mid[0][i], vec_mid[1][i]), vec_mid[2][i]);
        vec_dst[i][1] = vsubq_f16(vsubq_f16(vec_mid[1][i], vec_mid[2][i]), vec_mid[3][i]);
        vst1q_f16(dst_ptr + i * h_stride, vec_dst[i][0]);
        vst1q_f16(dst_ptr + i * h_stride + 8, vec_dst[i][1]);
    }
}

// B = [4  0  0  0  0  0] s0
//     [0 -4  4 -2  2  4] s1
//     [-5 -4 -4 -1 -1 0] s2
//     [0  1 -1  2 -2 -5] s3
//     [1  1  1  1  1  0] s4
//     [0  0  0  0  0  1] s5

// auto m0 = s0 * 4.f - s2 * 5.f + s4;
// auto m1 = s3 + s4 - (s1 + s2) * 4.f;
// auto m2 = (s1 - s2) * 4.f + s4 - s3;
// auto m3 = (s3 - s1) * 2.f + s4 - s2;
// auto m4 = s4 - s2 - (s3 - s1) * 2.f;
// auto m5 = s1 * 4.f - s3 * 5.f + s5
void SrcTransformInOne6x6Fp16(const void *src, void *dst, int w_stride, int h_stride) {
    float16x8_t vec_src[6][6];
    float16x8_t vec_mid[6][6];
    float16x8_t vec_dst[6][6];

    const __fp16 *src_f = reinterpret_cast<const __fp16 *>(src);
    __fp16 *dst_f = reinterpret_cast<__fp16 *>(dst);

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            vec_src[i][j] = vld1q_f16(src_f + i * h_stride + j * w_stride);
        }
    }

    for (int i = 0; i < 6; i++) {
        vec_mid[0][i] = vfmsq_n_f16(vfmaq_n_f16(vec_src[i][4], vec_src[i][0], 4.f), vec_src[i][2], 5.f);
        vec_mid[1][i] = vfmsq_n_f16(vaddq_f16(vec_src[i][3], vec_src[i][4]), vaddq_f16(vec_src[i][1], vec_src[i][2]), 4.0f);
        vec_mid[2][i] = vfmaq_n_f16(vsubq_f16(vec_src[i][4], vec_src[i][3]), vsubq_f16(vec_src[i][1], vec_src[i][2]), 4.0f);
        vec_mid[3][i] = vfmaq_n_f16(vsubq_f16(vec_src[i][4], vec_src[i][2]), vsubq_f16(vec_src[i][3], vec_src[i][1]), 2.0f);
        vec_mid[4][i] = vfmaq_n_f16(vsubq_f16(vec_src[i][4], vec_src[i][2]), vsubq_f16(vec_src[i][1], vec_src[i][3]), 2.0f);
        vec_mid[5][i] = vfmsq_n_f16(vfmaq_n_f16(vec_src[i][5], vec_src[i][1], 4.0f), vec_src[i][3], 5.0f);
    }

    for (int i = 0; i < 6; i++) {
        vec_dst[0][i] = vfmsq_n_f16(vfmaq_n_f16(vec_mid[i][4], vec_mid[i][0], 4.f), vec_mid[i][2], 5.f);
        vec_dst[1][i] = vfmsq_n_f16(vaddq_f16(vec_mid[i][3], vec_mid[i][4]), vaddq_f16(vec_mid[i][1], vec_mid[i][2]), 4.0f);
        vec_dst[2][i] = vfmaq_n_f16(vsubq_f16(vec_mid[i][4], vec_mid[i][3]), vsubq_f16(vec_mid[i][1], vec_mid[i][2]), 4.0f);
        vec_dst[3][i] = vfmaq_n_f16(vsubq_f16(vec_mid[i][4], vec_mid[i][2]), vsubq_f16(vec_mid[i][3], vec_mid[i][1]), 2.0f);
        vec_dst[4][i] = vfmaq_n_f16(vsubq_f16(vec_mid[i][4], vec_mid[i][2]), vsubq_f16(vec_mid[i][1], vec_mid[i][3]), 2.0f);
        vec_dst[5][i] = vfmsq_n_f16(vfmaq_n_f16(vec_mid[i][5], vec_mid[i][1], 4.0f), vec_mid[i][3], 5.0f);
        vst1q_f16(dst_f + 0   + 8 * i, vec_dst[0][i]);
        vst1q_f16(dst_f + 48  + 8 * i, vec_dst[1][i]);
        vst1q_f16(dst_f + 96  + 8 * i, vec_dst[2][i]);
        vst1q_f16(dst_f + 144 + 8 * i, vec_dst[3][i]);
        vst1q_f16(dst_f + 192 + 8 * i, vec_dst[4][i]);
        vst1q_f16(dst_f + 240 + 8 * i, vec_dst[5][i]);
    }
}

// A = [1  0  0  0]
//     [1  1  1  1]
//     [1 -1  1 -1]
//     [1  2  4  8]
//     [1 -2  4 -8]
//     [0  0  0  1]
// auto m0 = s0 + s1 + s2 + s3 + s4;
// auto m1 = s1 - s2 + (s3 - s4) * 2.f;
// auto m2 = s1 + s2 + (s3 + s4) * 4.f;
// auto m3 = s1 - s2 + (s3 - s4) * 8.f + s5;
void DstTransformInOne6x4Fp16(const void *src, void *dst, int w_stride, int h_stride, int ey) {
    float16x8_t vec_src[6][6];
    float16x8_t vec_mid[6][4];
    float16x8_t vec_dst[4][4];

    const __fp16 *src_f = reinterpret_cast<const __fp16 *>(src);
    __fp16 *dst_f = reinterpret_cast<__fp16 *>(dst);

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            vec_src[i][j] = vld1q_f16(src_f + i * 6 * w_stride + j * w_stride);
        }
    }

    for (int i = 0; i < 6; i++) {
        vec_mid[i][0] = vaddq_f16(vaddq_f16(vaddq_f16(vec_src[0][i], vec_src[1][i]), vaddq_f16(vec_src[2][i], vec_src[3][i])), vec_src[4][i]);
        vec_mid[i][1] = vfmaq_n_f16(vsubq_f16(vec_src[1][i], vec_src[2][i]), vsubq_f16(vec_src[3][i], vec_src[4][i]), 2.0f);
        vec_mid[i][2] = vfmaq_n_f16(vaddq_f16(vec_src[1][i], vec_src[2][i]), vaddq_f16(vec_src[3][i], vec_src[4][i]), 4.0f);
        vec_mid[i][3] = vaddq_f16(vfmaq_n_f16(vsubq_f16(vec_src[1][i], vec_src[2][i]), vsubq_f16(vec_src[3][i], vec_src[4][i]), 8.0f), vec_src[5][i]);
    }

    for (int i = 0; i < ey; i++) {
        vec_dst[i][0] = vaddq_f16(vaddq_f16(vaddq_f16(vec_mid[0][i], vec_mid[1][i]), vaddq_f16(vec_mid[2][i], vec_mid[3][i])), vec_mid[4][i]);
        vec_dst[i][1] = vfmaq_n_f16(vsubq_f16(vec_mid[1][i], vec_mid[2][i]), vsubq_f16(vec_mid[3][i], vec_mid[4][i]), 2.0f);
        vec_dst[i][2] = vfmaq_n_f16(vaddq_f16(vec_mid[1][i], vec_mid[2][i]), vaddq_f16(vec_mid[3][i], vec_mid[4][i]), 4.0f);
        vec_dst[i][3] = vaddq_f16(vfmaq_n_f16(vsubq_f16(vec_mid[1][i], vec_mid[2][i]), vsubq_f16(vec_mid[3][i], vec_mid[4][i]), 8.0f), vec_mid[5][i]);
        vst1q_f16(dst_f + i * h_stride, vec_dst[i][0]);
        vst1q_f16(dst_f + i * h_stride + 8, vec_dst[i][1]);
        vst1q_f16(dst_f + i * h_stride + 16, vec_dst[i][2]);
        vst1q_f16(dst_f + i * h_stride + 24, vec_dst[i][3]);
    }
}

}  // namespace TNN_NS
#endif
