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

#include "tnn/device/arm/acc/compute/winograd_function.h"
#include <cstring>
#include <memory>
#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif
#include "tnn/device/arm/acc/Float4.h"

namespace TNN_NS {

static inline void WeightTransform(const float *src, float *dst, int kernel_size, int unit, int in_channel,
                                   int out_channel, const float (*G)[3]) {
    float M[unit][3];
    float K_trans[unit * unit];

    int ic_4        = UP_DIV(in_channel, 4);
    int oc_4        = UP_DIV(out_channel, 4);
    int unit_stride = ic_4 * oc_4 * 4 * 4;
    int oc_stride   = ic_4 * 4 * 4;
    int ic_stride   = 4 * 4;

    for (int oc = 0; oc < out_channel; oc++) {
        int zo        = oc / 4;
        int ro        = oc % 4;
        float *dst_oz = dst + zo * oc_stride + ro;
        for (int ic = 0; ic < in_channel; ic++) {
            const float *src_z = src + (oc * in_channel + ic) * 3 * 3;
            const float *k0    = src_z;
            const float *k1    = k0 + 3;
            const float *k2    = k1 + 3;

            int zi = ic / 4;
            int ri = ic % 4;

            // M=G*g
            for (int i = 0; i < unit; i++) {
                M[i][0] = k0[0] * G[i][0] + k1[0] * G[i][1] + k2[0] * G[i][2];
                M[i][1] = k0[1] * G[i][0] + k1[1] * G[i][1] + k2[1] * G[i][2];
                M[i][2] = k0[2] * G[i][0] + k1[2] * G[i][1] + k2[2] * G[i][2];
            }

            // K_trans=M*GT
            for (int j = 0; j < unit; j++) {
                float *Mp = &M[j][0];
                for (int i = 0; i < unit; i++) {
                    K_trans[j * unit + i] = Mp[0] * G[i][0] + Mp[1] * G[i][1] + Mp[2] * G[i][2];
                }
            }

            auto dst_sz = dst_oz + zi * ic_stride + 4 * ri;

            for (int k = 0; k < unit * unit; k++) {
                *(dst_sz + k * unit_stride) = K_trans[k];
            }
        }
    }
}

void WeightTransform4x4(const float *src, float *dst, int kernel_size, int in_channel, int out_channel) {
    const float G[4][3] = {{1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}};

    WeightTransform(src, dst, kernel_size, 4, in_channel, out_channel, G);
}

void WeightTransform6x6(const float *src, float *dst, int kernel_size, int in_channel, int out_channel) {
    const float G[6][3] = {
        {1.0f / 4, 0.0f, 0.0f},           {-1.0f / 6, -1.0f / 6, -1.0f / 6}, {-1.0f / 6, 1.0f / 6, -1.0f / 6},
        {1.0f / 24, 1.0f / 12, 1.0f / 6}, {1.0f / 24, -1.0f / 12, 1.0f / 6}, {0.0f, 0.0f, 1.0f}};

    WeightTransform(src, dst, kernel_size, 6, in_channel, out_channel, G);
}

template <typename T>
void SrcTransformInOne4x4(const T *src, float *dst, int w_stride, int h_stride) {
    Float4 vec_src[4][4];
    Float4 vec_mid[4][4];
    Float4 vec_dst[4][4];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            vec_src[i][j] = Float4::load(src + i * h_stride + j * w_stride);
        }
    }

    for (int i = 0; i < 4; i++) {
        vec_mid[0][i] = vec_src[i][0] - vec_src[i][2];
        vec_mid[1][i] = vec_src[i][1] + vec_src[i][2];
        vec_mid[2][i] = vec_src[i][2] - vec_src[i][1];
        vec_mid[3][i] = vec_src[i][1] - vec_src[i][3];
    }

    for (int i = 0; i < 4; i++) {
        vec_dst[0][i] = vec_mid[i][0] - vec_mid[i][2];
        vec_dst[1][i] = vec_mid[i][1] + vec_mid[i][2];
        vec_dst[2][i] = vec_mid[i][2] - vec_mid[i][1];
        vec_dst[3][i] = vec_mid[i][1] - vec_mid[i][3];
        Float4::save(dst + 4 * i, vec_dst[0][i]);
        Float4::save(dst + 16 + 4 * i, vec_dst[1][i]);
        Float4::save(dst + 32 + 4 * i, vec_dst[2][i]);
        Float4::save(dst + 48 + 4 * i, vec_dst[3][i]);
    }
}

void SrcTransformInOne4x4Float(const void *src, void *dst, int w_stride, int h_stride) {
    SrcTransformInOne4x4<float>(reinterpret_cast<const float *>(src), reinterpret_cast<float *>(dst), w_stride,
                                h_stride);
}

void SrcTransformInOne4x4BFP16(const void *src, void *dst, int w_stride, int h_stride) {
    SrcTransformInOne4x4<bfp16_t>(reinterpret_cast<const bfp16_t *>(src), reinterpret_cast<float *>(dst), w_stride,
                                  h_stride);
}

// A = [1  0]
//     [1  1]
//     [1 -1]
//     [0 -1]
template <typename T>
void DstTransformInOne4x2(const float *src, T *dst, int w_stride, int h_stride, int ey) {
    Float4 vec_src[4][4];
    Float4 vec_mid[4][2];
    Float4 vec_dst[2][2];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            vec_src[i][j] = Float4::load(src + i * 4 * w_stride + j * w_stride);
        }
    }

    for (int i = 0; i < 4; i++) {
        vec_mid[i][0] = vec_src[0][i] + vec_src[1][i] + vec_src[2][i];
        vec_mid[i][1] = vec_src[1][i] - vec_src[2][i] - vec_src[3][i];
    }

    for (int i = 0; i < ey; i++) {
        vec_dst[i][0] = vec_mid[0][i] + vec_mid[1][i] + vec_mid[2][i];
        vec_dst[i][1] = vec_mid[1][i] - vec_mid[2][i] - vec_mid[3][i];
        Float4::save(dst + i * h_stride, vec_dst[i][0]);
        Float4::save(dst + i * h_stride + 4, vec_dst[i][1]);
    }
}

void DstTransformInOne4x2Float(const void *src, void *dst, int w_stride, int h_stride, int ey) {
    DstTransformInOne4x2<float>(reinterpret_cast<const float *>(src), reinterpret_cast<float *>(dst), w_stride,
                                h_stride, ey);
}

void DstTransformInOne4x2BFP16(const void *src, void *dst, int w_stride, int h_stride, int ey) {
    DstTransformInOne4x2<bfp16_t>(reinterpret_cast<const float *>(src), reinterpret_cast<bfp16_t *>(dst), w_stride,
                                  h_stride, ey);
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
template <typename T>
void SrcTransformInOne6x6(const T *src, float *dst, int w_stride, int h_stride) {
    Float4 vec_src[6][6];
    Float4 vec_mid[6][6];
    Float4 vec_dst[6][6];

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            vec_src[i][j] = Float4::load(src + i * h_stride + j * w_stride);
        }
    }

    for (int i = 0; i < 6; i++) {
        vec_mid[0][i] = vec_src[i][0] * 4.f - vec_src[i][2] * 5.f + vec_src[i][4];
        vec_mid[1][i] = vec_src[i][3] + vec_src[i][4] - (vec_src[i][1] + vec_src[i][2]) * 4.f;
        vec_mid[2][i] = (vec_src[i][1] - vec_src[i][2]) * 4.f + vec_src[i][4] - vec_src[i][3];
        vec_mid[3][i] = (vec_src[i][3] - vec_src[i][1]) * 2.f + vec_src[i][4] - vec_src[i][2];
        vec_mid[4][i] = vec_src[i][4] - vec_src[i][2] - (vec_src[i][3] - vec_src[i][1]) * 2.f;
        vec_mid[5][i] = vec_src[i][1] * 4.f - vec_src[i][3] * 5.f + vec_src[i][5];
    }

    for (int i = 0; i < 6; i++) {
        vec_dst[0][i] = vec_mid[i][0] * 4.f - vec_mid[i][2] * 5.f + vec_mid[i][4];
        vec_dst[1][i] = vec_mid[i][3] + vec_mid[i][4] - (vec_mid[i][1] + vec_mid[i][2]) * 4.f;
        vec_dst[2][i] = (vec_mid[i][1] - vec_mid[i][2]) * 4.f + vec_mid[i][4] - vec_mid[i][3];
        vec_dst[3][i] = (vec_mid[i][3] - vec_mid[i][1]) * 2.f + vec_mid[i][4] - vec_mid[i][2];
        vec_dst[4][i] = vec_mid[i][4] - vec_mid[i][2] - (vec_mid[i][3] - vec_mid[i][1]) * 2.f;
        vec_dst[5][i] = vec_mid[i][1] * 4.f - vec_mid[i][3] * 5.f + vec_mid[i][5];
        Float4::save(dst + 4 * i, vec_dst[0][i]);
        Float4::save(dst + 24 + 4 * i, vec_dst[1][i]);
        Float4::save(dst + 48 + 4 * i, vec_dst[2][i]);
        Float4::save(dst + 72 + 4 * i, vec_dst[3][i]);
        Float4::save(dst + 96 + 4 * i, vec_dst[4][i]);
        Float4::save(dst + 120 + 4 * i, vec_dst[5][i]);
    }
}

void SrcTransformInOne6x6Float(const void *src, void *dst, int w_stride, int h_stride) {
#if defined(__aarch64__)
    float32x2_t vec_src[6][6];
    float32x2_t vec_mid[6][6];
    float32x2_t vec_dst[6][6];

    auto src_f = reinterpret_cast<const float *>(src);
    auto dst_f = reinterpret_cast<float *>(dst);

    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                vec_src[i][j] = vld1_f32(src_f + k * 2 + i * h_stride + j * w_stride);
            }
        }

        for (int i = 0; i < 6; i++) {
            vec_mid[0][i] = vmls_n_f32(vmla_n_f32(vec_src[i][4], vec_src[i][0], 4.f), vec_src[i][2], 5.f);
            vec_mid[1][i] =
                vmls_n_f32(vadd_f32(vec_src[i][3], vec_src[i][4]), vadd_f32(vec_src[i][1], vec_src[i][2]), 4.0f);
            vec_mid[2][i] =
                vmla_n_f32(vsub_f32(vec_src[i][4], vec_src[i][3]), vsub_f32(vec_src[i][1], vec_src[i][2]), 4.0f);
            vec_mid[3][i] =
                vmla_n_f32(vsub_f32(vec_src[i][4], vec_src[i][2]), vsub_f32(vec_src[i][3], vec_src[i][1]), 2.0f);
            vec_mid[4][i] =
                vmla_n_f32(vsub_f32(vec_src[i][4], vec_src[i][2]), vsub_f32(vec_src[i][1], vec_src[i][3]), 2.0f);
            vec_mid[5][i] = vmls_n_f32(vmla_n_f32(vec_src[i][5], vec_src[i][1], 4.0f), vec_src[i][3], 5.0f);
        }

        for (int i = 0; i < 6; i++) {
            vec_dst[0][i] = vmls_n_f32(vmla_n_f32(vec_mid[i][4], vec_mid[i][0], 4.f), vec_mid[i][2], 5.f);
            vec_dst[1][i] =
                vmls_n_f32(vadd_f32(vec_mid[i][3], vec_mid[i][4]), vadd_f32(vec_mid[i][1], vec_mid[i][2]), 4.0f);
            vec_dst[2][i] =
                vmla_n_f32(vsub_f32(vec_mid[i][4], vec_mid[i][3]), vsub_f32(vec_mid[i][1], vec_mid[i][2]), 4.0f);
            vec_dst[3][i] =
                vmla_n_f32(vsub_f32(vec_mid[i][4], vec_mid[i][2]), vsub_f32(vec_mid[i][3], vec_mid[i][1]), 2.0f);
            vec_dst[4][i] =
                vmla_n_f32(vsub_f32(vec_mid[i][4], vec_mid[i][2]), vsub_f32(vec_mid[i][1], vec_mid[i][3]), 2.0f);
            vec_dst[5][i] = vmls_n_f32(vmla_n_f32(vec_mid[i][5], vec_mid[i][1], 4.0f), vec_mid[i][3], 5.0f);
            vst1_f32(dst_f + k * 2 + 4 * i, vec_dst[0][i]);
            vst1_f32(dst_f + k * 2 + 24 + 4 * i, vec_dst[1][i]);
            vst1_f32(dst_f + k * 2 + 48 + 4 * i, vec_dst[2][i]);
            vst1_f32(dst_f + k * 2 + 72 + 4 * i, vec_dst[3][i]);
            vst1_f32(dst_f + k * 2 + 96 + 4 * i, vec_dst[4][i]);
            vst1_f32(dst_f + k * 2 + 120 + 4 * i, vec_dst[5][i]);
        }
    }
#else
    SrcTransformInOne6x6<float>(reinterpret_cast<const float *>(src), reinterpret_cast<float *>(dst), w_stride,
                                h_stride);
#endif
}

void SrcTransformInOne6x6BFP16(const void *src, void *dst, int w_stride, int h_stride) {
    SrcTransformInOne6x6<bfp16_t>(reinterpret_cast<const bfp16_t *>(src), reinterpret_cast<float *>(dst), w_stride,
                                  h_stride);
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
template <typename T>
void DstTransformInOne6x4(const float *src, T *dst, int w_stride, int h_stride, int ey) {
    Float4 vec_src[6][6];
    Float4 vec_mid[6][4];
    Float4 vec_dst[4][4];

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            vec_src[i][j] = Float4::load(src + i * 6 * w_stride + j * w_stride);
        }
    }

    for (int i = 0; i < 6; i++) {
        vec_mid[i][0] = vec_src[0][i] + vec_src[1][i] + vec_src[2][i] + vec_src[3][i] + vec_src[4][i];
        vec_mid[i][1] = vec_src[1][i] - vec_src[2][i] + (vec_src[3][i] - vec_src[4][i]) * 2.f;
        vec_mid[i][2] = vec_src[1][i] + vec_src[2][i] + (vec_src[3][i] + vec_src[4][i]) * 4.f;
        vec_mid[i][3] = vec_src[1][i] - vec_src[2][i] + (vec_src[3][i] - vec_src[4][i]) * 8.f + vec_src[5][i];
    }

    for (int i = 0; i < ey; i++) {
        vec_dst[i][0] = vec_mid[0][i] + vec_mid[1][i] + vec_mid[2][i] + vec_mid[3][i] + vec_mid[4][i];
        vec_dst[i][1] = vec_mid[1][i] - vec_mid[2][i] + (vec_mid[3][i] - vec_mid[4][i]) * 2.f;
        vec_dst[i][2] = vec_mid[1][i] + vec_mid[2][i] + (vec_mid[3][i] + vec_mid[4][i]) * 4.f;
        vec_dst[i][3] = vec_mid[1][i] - vec_mid[2][i] + (vec_mid[3][i] - vec_mid[4][i]) * 8.f + vec_mid[5][i];
        Float4::save(dst + i * h_stride, vec_dst[i][0]);
        Float4::save(dst + i * h_stride + 4, vec_dst[i][1]);
        Float4::save(dst + i * h_stride + 8, vec_dst[i][2]);
        Float4::save(dst + i * h_stride + 12, vec_dst[i][3]);
    }
}

void DstTransformInOne6x4Float(const void *src, void *dst, int w_stride, int h_stride, int ey) {
#if defined(__aarch64__)
    float32x2_t vec_src[6][6];
    float32x2_t vec_mid[6][4];
    float32x2_t vec_dst[4][4];

    auto src_f = reinterpret_cast<const float *>(src);
    auto dst_f = reinterpret_cast<float *>(dst);

    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                vec_src[i][j] = vld1_f32(src_f + k * 2 + i * 6 * w_stride + j * w_stride);
            }
        }

        for (int i = 0; i < 6; i++) {
            vec_mid[i][0] =
                vadd_f32(vadd_f32(vadd_f32(vec_src[0][i], vec_src[1][i]), vadd_f32(vec_src[2][i], vec_src[3][i])),
                         vec_src[4][i]);
            vec_mid[i][1] =
                vmla_n_f32(vsub_f32(vec_src[1][i], vec_src[2][i]), vsub_f32(vec_src[3][i], vec_src[4][i]), 2.0f);
            vec_mid[i][2] =
                vmla_n_f32(vadd_f32(vec_src[1][i], vec_src[2][i]), vadd_f32(vec_src[3][i], vec_src[4][i]), 4.0f);
            vec_mid[i][3] = vadd_f32(
                vmla_n_f32(vsub_f32(vec_src[1][i], vec_src[2][i]), vsub_f32(vec_src[3][i], vec_src[4][i]), 8.0f),
                vec_src[5][i]);
        }

        for (int i = 0; i < ey; i++) {
            vec_dst[i][0] =
                vadd_f32(vadd_f32(vadd_f32(vec_mid[0][i], vec_mid[1][i]), vadd_f32(vec_mid[2][i], vec_mid[3][i])),
                         vec_mid[4][i]);
            vec_dst[i][1] =
                vmla_n_f32(vsub_f32(vec_mid[1][i], vec_mid[2][i]), vsub_f32(vec_mid[3][i], vec_mid[4][i]), 2.0f);
            vec_dst[i][2] =
                vmla_n_f32(vadd_f32(vec_mid[1][i], vec_mid[2][i]), vadd_f32(vec_mid[3][i], vec_mid[4][i]), 4.0f);
            vec_dst[i][3] = vadd_f32(
                vmla_n_f32(vsub_f32(vec_mid[1][i], vec_mid[2][i]), vsub_f32(vec_mid[3][i], vec_mid[4][i]), 8.0f),
                vec_mid[5][i]);
            vst1_f32(dst_f + k * 2 + i * h_stride, vec_dst[i][0]);
            vst1_f32(dst_f + k * 2 + i * h_stride + 4, vec_dst[i][1]);
            vst1_f32(dst_f + k * 2 + i * h_stride + 8, vec_dst[i][2]);
            vst1_f32(dst_f + k * 2 + i * h_stride + 12, vec_dst[i][3]);
        }
    }
#else
    DstTransformInOne6x4<float>(reinterpret_cast<const float *>(src), reinterpret_cast<float *>(dst), w_stride,
                                h_stride, ey);
#endif
}

void DstTransformInOne6x4BFP16(const void *src, void *dst, int w_stride, int h_stride, int ey) {
    DstTransformInOne6x4<bfp16_t>(reinterpret_cast<const float *>(src), reinterpret_cast<bfp16_t *>(dst), w_stride,
                                  h_stride, ey);
}

}  // namespace TNN_NS
