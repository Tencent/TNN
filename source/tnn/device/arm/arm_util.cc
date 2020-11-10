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

#include "tnn/device/arm/arm_util.h"

#include <type_traits>

#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

#include "tnn/core/macro.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

#ifdef TNN_USE_NEON
int PackNeon(float *dst, const float *src, size_t hw, size_t channel) {
    for (int c = 0; c < channel; c += 4) {
        auto src_c = src + c * hw;
        auto dst_c = dst + c * hw;
        for (int cur_hw = 0; cur_hw < hw; cur_hw += 4) {
            float32x4x4_t v;
            v.val[0] = vld1q_f32(src_c + cur_hw);
            v.val[1] = vld1q_f32(src_c + cur_hw + hw * 1);
            v.val[2] = vld1q_f32(src_c + cur_hw + hw * 2);
            v.val[3] = vld1q_f32(src_c + cur_hw + hw * 3);
            vst4q_f32(dst_c + cur_hw * 4, v);
        }
    }

    return 0;
}
int PackNeonC3(float *dst, const float *src, size_t hw, size_t channel) {
    auto src0 = src;
    auto src1 = src + hw;
    auto src2 = src + hw * 2;
    for (int cur_hw = 0; cur_hw < hw; cur_hw += 4) {
        float32x4x4_t v;
        v.val[0] = vld1q_f32(src0 + cur_hw);
        v.val[1] = vld1q_f32(src1 + cur_hw);
        v.val[2] = vld1q_f32(src2 + cur_hw);
        v.val[3] = vdupq_n_f32(0);
        vst4q_f32(dst + cur_hw * 4, v);
    }

    return 0;
}
int PackNeonNHWC(float *dst, const float *src, size_t hw, size_t channel) {
    if ((hw == 1) && (channel % 4 == 0)) {
        memcpy(dst, src, hw * channel * sizeof(float));
        return 0;
    }

    auto cc = (channel>>2<<2);
    float32x4_t v;
    for (int c = 0; c < cc; c += 4) {
        auto src_c = src + c;
        auto dst_c = dst + c * hw;
        for (int cur_hw = 0; cur_hw < hw; ++cur_hw) {
            v = vld1q_f32(src_c);
            vst1q_f32(dst_c, v);
            src_c += channel;
            dst_c += 4;
        }
    }

    int remain = channel % 4;
    if (remain) {
        auto src_c = src + cc;
        auto dst_c = dst + cc * hw;
        for (int cur_hw = 0; cur_hw < hw; ++cur_hw) {
            v = vdupq_n_f32(0);
            for (int r = 0; r < remain; ++r)
                v[r] = *(src_c + r);
            vst1q_f32(dst_c, v);
            src_c += channel;
            dst_c += 4;
        }
    }

    return 0;
}
int UnpackNeon(float *dst, const float *src, size_t hw, size_t channel) {
    float32x4x4_t v;
    for (int c = 0; c < channel; c += 4) {
        auto src_c = src + c * hw;
        auto dst_c = dst + c * hw;
        for (int cur_hw = 0; cur_hw < hw; cur_hw += 4) {
            v = vld4q_f32(src_c + cur_hw * 4);
            vst1q_f32(dst_c + cur_hw, v.val[0]);
            vst1q_f32(dst_c + cur_hw + hw * 1, v.val[1]);
            vst1q_f32(dst_c + cur_hw + hw * 2, v.val[2]);
            vst1q_f32(dst_c + cur_hw + hw * 3, v.val[3]);
        }
    }

    return 0;
}
int UnpackNeonNHWC(float *dst, const float *src, size_t hw, size_t channel) {
    if ((hw == 1) && (channel % 4 == 0)) {
        memcpy(dst, src, hw * channel * sizeof(float));
        return 0;
    }

    auto cc = (channel>>2<<2);
    float32x4_t v;
    for (int c = 0; c < cc; c += 4) {
        auto dst_c = dst + c;
        auto src_c = src + c * hw;
        for (int cur_hw = 0; cur_hw < hw; ++cur_hw) {
            v = vld1q_f32(src_c);
            vst1q_f32(dst_c, v);
            src_c += 4;
            dst_c += channel;
        }
    }

    int remain = channel % 4;
    if (remain) {
        auto dst_c = dst + cc;
        auto src_c = src + cc * hw;
        for (int cur_hw = 0; cur_hw < hw; ++cur_hw) {
            v = vld1q_f32(src_c);
            for (int r = 0; r < remain; ++r)
                *(dst_c + r) = v[r];
            src_c += 4;
            dst_c += channel;
        }
    }

    return 0;
}
#endif

char* GetBlobHandlePtr(BlobHandle handle) {
    return reinterpret_cast<char *>(handle.base) + handle.bytes_offset; 
}

template <typename Tin, typename Tout>
int PackC4(Tout *dst, const Tin *src, size_t hw, size_t channel) {
#ifdef TNN_USE_NEON
    if (std::is_same<Tin, float>::value && std::is_same<Tout, float>::value) {
        if (channel % 4 == 0 && hw % 4 == 0) {
            return PackNeon((float *)dst, (const float *)src, hw, channel);
        } else if (channel == 3 && hw % 4 == 0) {
            return PackNeonC3((float *)dst, (const float *)src, hw, channel);
        }
    }
#endif
    int c, cur_hw;
    int idx = 0;
    memset(dst, 0, hw * UP_DIV(channel, 4) * 4 * sizeof(Tout));
    for (c = 0; c < channel; ++c) {
        int plane      = c / 4;
        auto *dstPlane = plane * hw * 4 + dst;
        int offset     = c % 4;
        for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
            dstPlane[4 * cur_hw + offset] = src[idx++];
        }
    }

    return 0;
}

template int PackC4(float *dst, const float *src, size_t hw, size_t channel);
template int PackC4(bfp16_t *dst, const float *src, size_t hw, size_t channel);
template int PackC4(float *dst, const bfp16_t *src, size_t hw, size_t channel);
template int PackC4(bfp16_t *dst, const bfp16_t *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int PackC4FromNHWC(Tout *dst, const Tin *src, size_t hw, size_t channel) {
#ifdef TNN_USE_NEON
    if (std::is_same<Tin, float>::value && std::is_same<Tout, float>::value) {
        return PackNeonNHWC((float *)dst, (const float *)src, hw, channel);
    }
#endif
    int c, cur_hw;
    int idx = 0;
    memset(dst, 0, hw * UP_DIV(channel, 4) * 4 * sizeof(Tout));
    for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
        for (c = 0; c < channel; ++c) {
            int plane      = c / 4;
            auto *dstPlane = plane * hw * 4 + dst;
            int offset     = c % 4;
            dstPlane[4 * cur_hw + offset] = src[idx++];
        }
    }

    return 0;
}

template int PackC4FromNHWC(float *dst, const float *src, size_t hw, size_t channel);
template int PackC4FromNHWC(bfp16_t *dst, const float *src, size_t hw, size_t channel);
template int PackC4FromNHWC(float *dst, const bfp16_t *src, size_t hw, size_t channel);
template int PackC4FromNHWC(bfp16_t *dst, const bfp16_t *src, size_t hw, size_t channel);

int PackCAndQuant(int8_t *dst, const float *src, size_t hw, size_t channel, float *scale) {
    int idx  = 0;
    int c_r4 = ROUND_UP(channel, 4);
    memset(dst, 0, hw * c_r4);
    for (int c = 0; c < channel; ++c) {
        int8_t *dst_c = dst + c;
        for (int cur_hw = 0; cur_hw < hw; ++cur_hw) {
            dst_c[cur_hw * c_r4] = float2int8(src[idx++] * scale[c]);
        }
    }

    return 0;
}

template <typename Tin, typename Tout>
int UnpackC4(Tout *dst, const Tin *src, size_t hw, size_t channel) {
#ifdef TNN_USE_NEON
    if (std::is_same<Tin, float>::value && std::is_same<Tout, float>::value) {
        if (channel % 4 == 0 && hw % 4 == 0) {
            return UnpackNeon((float *)dst, (const float *)src, hw, channel);
        }
    }
#endif
    int cur_hw;
    int c;
    int idx = 0;
    for (c = 0; c < channel; ++c) {
        int plane         = c / 4;
        const auto *src_c = plane * hw * 4 + src;
        int offset        = c % 4;
        for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
            dst[idx++] = src_c[4 * cur_hw + offset];
        }
    }
    return 0;
}

template int UnpackC4(float *dst, const float *src, size_t hw, size_t channel);
template int UnpackC4(float *dst, const bfp16_t *src, size_t hw, size_t channel);
template int UnpackC4(bfp16_t *dst, const float *src, size_t hw, size_t channel);
template int UnpackC4(bfp16_t *dst, const bfp16_t *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int UnpackC4ToNHWC(Tout *dst, const Tin *src, size_t hw, size_t channel) {
#ifdef TNN_USE_NEON
    if (std::is_same<Tin, float>::value && std::is_same<Tout, float>::value) {
        return UnpackNeonNHWC((float *)dst, (const float *)src, hw, channel);
    }
#endif
    int cur_hw;
    int c;
    int idx = 0;
    for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
        for (c = 0; c < channel; ++c) {
            int plane         = c / 4;
            const auto *src_c = plane * hw * 4 + src;
            int offset        = c % 4;
            dst[idx++] = src_c[4 * cur_hw + offset];
        }
    }
    return 0;
}

template int UnpackC4ToNHWC(float *dst, const float *src, size_t hw, size_t channel);
template int UnpackC4ToNHWC(float *dst, const bfp16_t *src, size_t hw, size_t channel);
template int UnpackC4ToNHWC(bfp16_t *dst, const float *src, size_t hw, size_t channel);
template int UnpackC4ToNHWC(bfp16_t *dst, const bfp16_t *src, size_t hw, size_t channel);

int UnpackAndDequant(float *dst, const int8_t *src, size_t hw, size_t channel, float *scale, float *bias) {
    int cur_hw;
    int c;
    int idx  = 0;
    int c_r4 = ROUND_UP(channel, 4);
    for (c = 0; c < channel; ++c) {
        auto *src_c = src + c;
        for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
            dst[idx++] = src_c[c_r4 * cur_hw] * scale[c] + bias[c];
        }
    }
    return 0;
}

int UnpackC4WithStride(float *dst, const float *src, size_t ih, size_t iw, size_t c_step, size_t w_step,
                       size_t channel) {
    int c;

    for (c = 0; c < channel; ++c) {
        int plane          = c / 4;
        const float *src_c = plane * ih * iw * 4 + src;
        float *dst_c_start = dst + c * c_step;
        for (int h = 0; h < ih; h++) {
            float *dst_x_start = dst_c_start + h * w_step;
            int offset         = c % 4;
            for (int w = 0; w < iw; w++) {
                *dst_x_start++ = src_c[4 * (h * iw + w) + offset];
            }
        }
    }

    return 0;
}

#define ConvertWeightsPreparation                                        \
    const int goc       = output_channel / group;                        \
    const int gic       = input_channel / group;                         \
    const int goc_4     = (goc + 3) / 4;                                 \
    const int gic_4     = (gic + 3) / 4;                                 \
    const int src_count = group * goc * gic * height * width;            \
    int src_index = 0;
// to   [g][o/4][i/4][h][w][16]
// from [g][o][i][h][w]
template <typename T>
int ConvertWeightsFromGOIHWToGOIHW16(T *src, T *dst, int group, int input_channel, int output_channel, int height,
                                     int width) {
    ConvertWeightsPreparation;

    for (int g = 0; g < group; g++) {
        auto g_dst = dst + g * goc_4 * gic_4 * height * width * 16;  // g
        for (int o = 0; o < goc; o++) {
            auto zo = o / 4, ro = o % 4;
            auto o_dst = g_dst + zo * gic_4 * height * width * 16 + ro;  // o/4 x 4
            for (int i = 0; i < gic; i++) {
                auto zi = i / 4, ri = i % 4;
                auto i_dst = o_dst + zi * height * width * 16 + ri * 4;  // i/4 x 4
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        // to   [g][o/4][i/4][h][w][16]
                        // from [g][o][i][h][w]
                        if (src_index < src_count) {
                            i_dst[(h * width + w) * 16] = src[src_index++];
                        } else {
                            i_dst[(h * width + w) * 16] = 0;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

template int ConvertWeightsFromGOIHWToGOIHW16(float *src, float *dst, int group, int input_channel, int output_channel,
                                              int height, int width);

// to   [g][o/4][h][w][i/4][16]
// from [g][i][o][h][w]
template <typename T>
int ConvertWeightsFromGIOHWToGOHWI16(T *src, T *dst, int group, int input_channel, int output_channel, int height,
                                     int width) {
    ConvertWeightsPreparation;

    for (int g = 0; g < group; g++) {
        auto g_dst = dst + g * goc_4 * gic_4 * height * width * 16;  // g
        for (int i = 0; i < gic; i++) {
            auto zi = i / 4, ri = i % 4;
            auto i_dst = g_dst + zi * 16 + ri * 4;
            for (int o = 0; o < goc; o++) {
                auto zo = o / 4, ro = o % 4;
                auto o_dst = i_dst + zo * height * width * gic_4 * 16 + ro;
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        if (src_index < src_count) {
                            o_dst[(h * width + w) * gic_4 * 16] = src[src_index++];
                        } else {
                            o_dst[(h * width + w) * gic_4 * 16] = 0;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

template int ConvertWeightsFromGIOHWToGOHWI16(float *src, float *dst, int group, int input_channel, int output_channel,
                                              int height, int width);

template <typename T>
int ConvertWeightsC4ToC8(T *weight, int ic, int oc) {
    int ic4 = UP_DIV(ic, 4), oc4 = UP_DIV(oc, 4);
    RawBuffer temp(ic4 * oc4 * 16 * sizeof(T));

    for (int o = 0; o < oc4 * 4; o++) {
        for (int i = 0; i < ic4 * 4; i++) {
            int d_zo = o / 8, d_ro = o % 8, d_zi = i / 4, d_ri = i % 4;
            int s_zo = o / 4, s_ro = o % 4, s_zi = i / 4, s_ri = i % 4;
            int o_offset = d_zo * ic4 * 32 + d_zi * 32 + d_ri * 8 + d_ro;
            int i_offset = s_zo * ic4 * 16 + s_zi * 16 + s_ri * 4 + s_ro;

            temp.force_to<T *>()[o_offset] = weight[i_offset];
        }
    }
    memcpy(weight, temp.force_to<T *>(), temp.GetBytesSize());
    return 0;
}
template int ConvertWeightsC4ToC8(float *weight, int ic, int oc);
template int ConvertWeightsC4ToC8(bfp16_t *weight, int ic, int oc);

// from [o][i][h][w]
// to armv8: [o/4][h][w][i/16][o4][i16]
// to armv7: [o/4][h][w][i/8][o2][i2][o2][i4]
int PackINT8Weight(int8_t *src, int8_t *dst, int group, int input_channel, int output_channel, int height, int width) {
    const int oc_4        = (output_channel + 3) / 4;
    const int ic_calc     = input_channel < 4 ? input_channel : ROUND_UP(input_channel, 4);
    const int crs_round16 = ROUND_UP(ic_calc * height * width, 16);
    memset(dst, 0, oc_4 * 4 * crs_round16);
    for (int o = 0; o < output_channel; o++) {
        auto zo = o / 4, ro = o % 4;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int i = 0; i < input_channel; i++) {
#if !defined(__aarch64__) && defined(TNN_USE_NEON)
                    // to armv7: [o/4][h][w][i/8][o2][i2][o2][i4]
                    // so dirty and difficult to read but best for kernel
                    auto zro = ro / 2, rro = ro % 2;
                    auto o_dst = dst + zo * 4 * crs_round16 + zro * 16 + rro * 4;
                    auto zi    = ((h * width + w) * ic_calc + i) / 8;
                    auto ri    = ((h * width + w) * ic_calc + i) % 8;
                    auto zri = ri / 4, rri = ri % 4;
                    o_dst[zi * 8 * 4 + zri * 4 * 2 + rri] =
                        src[o * input_channel * height * width + i * height * width + h * width + w];
#else
                    // to armv8: [o/4][h][w][i/16][o4][i16]
                    auto o_dst = dst + zo * 4 * crs_round16 + ro * 16;
                    auto ri    = ((h * width + w) * ic_calc + i) % 16;
                    auto zi    = ((h * width + w) * ic_calc + i) / 16;
                    o_dst[zi * 16 * 4 + ri] =
                        src[o * input_channel * height * width + i * height * width + h * width + w];
#endif
                }
            }
        }
    }
    return 0;
}

// to   [g][o/4][h][w][12]
// from [g][o][i][h][w]
template <typename T>
int ConvertWeightsFromOI3HWToOHW12(T *src, T *dst, int input_channel, int output_channel, int height, int width) {
    const int oc_4      = (output_channel + 3) / 4;
    const int ic_4      = (input_channel + 3) / 4;
    const int src_count = output_channel * input_channel * height * width;

    int src_index = 0;

    for (int o = 0; o < output_channel; o++) {
        auto zo = o / 4, ro = o % 4;
        auto o_dst = dst + zo * height * width * 12 + ro;  // o/4 x 4
        for (int i = 0; i < input_channel; i++) {
            auto zi = i / 3, ri = i % 3;
            auto i_dst = o_dst + zi * height * width * 12 + ri * 4;  // i/4 x 4
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // to   [g][o/4][h][w][12]
                    // from [g][o][i][h][w]
                    if (src_index < src_count) {
                        i_dst[(h * width + w) * 12] = src[src_index++];
                    } else {
                        i_dst[(h * width + w) * 12] = 0;
                    }
                }
            }
        }
    }

    return 0;
}

template int ConvertWeightsFromOI3HWToOHW12(float *src, float *dst, int input_channel, int output_channel, int height,
                                            int width);

//float
//     r = 1.164 * (y - 16) + 1.596 * (v - 128);
//     g = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128);
//     b = 1.164 * (y - 16) + 2.018 * (u - 128);
//int 16
//     r = (74 *y - 1135 + 102 * vv ) >> 6
//     g = (74 *y - 1135 - 52 * vv - 25 * uu ) >> 6
//     b = (74 *y - 1135 + 129 * uu ) >> 6
void NV12ToBGR(const unsigned char* nv12, unsigned char* bgr, int h, int w) {
#ifndef TNN_USE_NEON
    return NaiveYUVToBGROrBGRA(nv12, bgr, 3, h, w, true);
#else
    const unsigned char* yptr  = nv12;
    const unsigned char* vuptr = nv12 + w * h;

    for (int y = 0; y < h; y += 2) {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* rgb0 = bgr;
        unsigned char* rgb1 = bgr + w * 3;
#if __aarch64__
        int64_t nn = w >> 3;
        int remain = w - (nn << 3);

        int16x8_t _q1135 = vdupq_n_s16(1135);
        int8x8_t _v74    = vdup_n_s8(74);
        int8x8_t _v128   = vdup_n_s8(int8_t(128));
        int8x8_t _v102   = vdup_n_s8(102);
        int8x8_t _v52    = vdup_n_s8(52);
        int8x8_t _v25    = vdup_n_s8(25);
        // use 127 instead of 129 to prevent char overflow, add another 2 in asm
        int8x8_t _v127   = vdup_n_s8(127);
        // saturate uv to 240 to avoid b overflow
        uint8x8_t _v240  = vdup_n_u8(240);

        if (nn > 0) {
            asm volatile(
                "prfm  pldl1strm, [%[_vu], #128]    \n\t"
                "ld1   {v2.8b},   [%[_vu]], #8      \n\t"
                "cmhi  v12.8b, v2.8b, %[_v240].8b   \n\t"
                "bsl   v12.8b, %[_v240].8b, v2.8b   \n\t"
                "sub   v2.8b, v12.8b, %[_v128].8b   \n\t"
                "0:                                 \n\t"
                "prfm  pldl1strm, [%[_y0], #128]    \n\t"
                "ld1   {v0.8b},   [%[_y0]], #8      \n\t"
                "prfm  pldl1strm, [%[_y1], #128]    \n\t"
                "ld1   {v1.8b},   [%[_y1]], #8      \n\t"
                "umull v28.8h, v0.8b,  %[_v74].8b   \n\t"
                "sub   v28.8h, v28.8h, %[_q1135].8h \n\t"   // v28 -> b0
                "orr   v3.8b,  v2.8b,  v2.8b        \n\t"
                "umull v29.8h, v1.8b,  %[_v74].8b   \n\t"
                "sub   v29.8h, v29.8h, %[_q1135].8h \n\t"   // v29 -> b1
                "orr   v9.16b, v28.16b, v28.16b     \n\t"   // v9  -> g0
                "trn1  v31.8b, v2.8b, v3.8b         \n\t"   // u
                "trn2  v30.8b, v2.8b, v3.8b         \n\t"   // v
                "orr   v11.16b, v29.16b, v29.16b    \n\t"   // v11 -> g1
                "sshll v27.8h, v31.8b, #1           \n\t"
                "smlsl v9.8h,  v30.8b, %[_v52].8b   \n\t"
                "orr   v8.16b, v28.16b, v28.16b     \n\t"   // v8  -> r0
                "smlsl v11.8h, v30.8b, %[_v52].8b   \n\t"
                "orr   v10.16b, v29.16b, v29.16b    \n\t"   // v10 -> r1
                "smlal v8.8h,  v30.8b, %[_v102].8b  \n\t"
                "smlal v28.8h, v31.8b, %[_v127].8b  \n\t"
                "smlal v10.8h, v30.8b, %[_v102].8b  \n\t"
                "add   v28.8h, v28.8h, v27.8h       \n\t"
                "smlsl v9.8h,  v31.8b, %[_v25].8b   \n\t"
                "smlal v29.8h, v31.8b, %[_v127].8b  \n\t"
                "smlsl v11.8h, v31.8b, %[_v25].8b   \n\t"
                "add   v29.8h, v29.8h, v27.8h       \n\t"
                "sqshrun v26.8b, v8.8h,  #6         \n\t"   // v24-v26: b0g0r0
                "sqshrun v24.8b, v28.8h, #6         \n\t"
                "sqshrun v6.8b,  v10.8h, #6         \n\t"
                "sqshrun v25.8b, v9.8h,  #6         \n\t"   // v4-v6: b1g1r1
                "sqshrun v4.8b,  v29.8h, #6         \n\t"
                "sqshrun v5.8b,  v11.8h, #6         \n\t"
                "prfm pldl1strm, [%[_vu], #128]     \n\t"
                "ld1 {v2.8b},    [%[_vu]], #8       \n\t"
                "subs %[_nn], %[_nn], #1            \n\t"
                "prfm pstl1strm, [%[_r0]]           \n\t"
                "st3 {v24.8b-v26.8b}, [%[_r0]], #24 \n\t"
                "cmhi  v12.8b, v2.8b, %[_v240].8b   \n\t"
                "bsl   v12.8b, %[_v240].8b, v2.8b   \n\t"
                "sub   v2.8b, v12.8b, %[_v128].8b   \n\t"
                "prfm pstl1strm, [%[_r1]]           \n\t"
                "st3 {v4.8b-v6.8b},   [%[_r1]], #24 \n\t"
                "bne 0b                             \n\t"
                "sub %[_vu], %[_vu], #8             \n\t"

                : [_nn]"+r"(nn),
                  [_y0]"+r"(yptr0),
                  [_y1]"+r"(yptr1),
                  [_vu]"+r"(vuptr),
                  [_r0]"+r"(rgb0),
                  [_r1]"+r"(rgb1)
                : [_v128]"w"(_v128),
                  [_v102]"w"(_v102),
                  [_v52]"w"(_v52),
                  [_v25]"w"(_v25),
                  [_v127]"w"(_v127),
                  [_q1135]"w"(_q1135),
                  [_v74]"w"(_v74),
                  [_v240]"w"(_v240)
                : "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8",
                  "v9", "v10", "v11", "v12", "v24", "v25", "v26","v27", "v28", "v29", "v30", "v31"
            );
        }
#else
        int nn         = w >> 3;
        int remain     = w - (nn << 3);
        short _s1135   = 1135;
        int8x8_t _v74  = vdup_n_s8(74);
        int8x8_t _v128 = vdup_n_s8(int8_t(128));
        // to much input w cause compile error, merge to one
        int8x8_t _vuvfilter = {102, 52, 25, 127, 0, 0, 0, 0};
        // saturate uv to 240 to avoid b overflow
        uint8x8_t _v240     = vdup_n_u8(240);

        if (nn > 0) {
            asm volatile(
                "pld        [%[_vu], #128]      \n"
                "vld1.u8    {d2}, [%[_vu]]!     \n"
                "vcgt.u8    d27, d2, %[_v240]   \n"
                "vbsl.u8    d27,  %[_v240], d2  \n"
                "vsub.u8    d2, d27, %[_v128]   \n"
                "vmov.s8    d10, %[_filt]       \n"
                "vdup.8     d11, d10[1]         \n"   // v52
                "vdup.8     d12, d10[2]         \n"   // v25
                "vdup.8     d13, d10[3]         \n"   // v127
                "vdup.16    q7,  %[_s1135]      \n"   // q1135
                "vdup.8     d10, d10[0]         \n"   // v102
                "0:                             \n"
                "pld        [%[_y0], #128]      \n"
                "vld1.u8    {d0}, [%[_y0]]!     \n"
                "pld        [%[_y1], #128]      \n"
                "vld1.u8    {d1}, [%[_y1]]!     \n"
                "vmull.u8   q2, d0, %[_v74]     \n"
                "vorr       d3, d2, d2          \n"
                "vsub.s16   q2, q2, q7          \n"   // q2  -> b0
                "vmull.u8   q3, d1, %[_v74]     \n"
                "vorr       q9, q2, q2          \n"   // q9  -> g0
                "vsub.s16   q3, q3, q7          \n"   // q3  -> b1
                "vtrn.s8    d3, d2              \n"   // d3 -> u, d2 -> v
                "vorr       q11, q3, q3         \n"   // q11 -> g1
                "vshll.s8   q4, d3, #1          \n"
                "vmlsl.s8   q9, d2, d11         \n"
                "vorr       q8, q2, q2          \n"   // q8  -> r0
                "vmlsl.s8   q11, d2, d11        \n"
                "vorr       q10, q3, q3         \n"   // q10 -> r1
                "vmlal.s8   q8, d2, d10         \n"
                "vmlal.s8   q2, d3, d13         \n"
                "vmlal.s8   q10, d2, d10        \n"
                "vadd.s16   q2, q2, q4          \n"
                "vmlsl.s8   q9, d3, d12         \n"
                "vmlal.s8   q3, d3, d13         \n"
                "vmlsl.s8   q11,d3, d12         \n"
                "vadd.s16   q3, q3, q4          \n"
                "vqshrun.s16 d26, q8, #6        \n"   // d24-d26: b0g0r0
                "vqshrun.s16 d24, q2, #6        \n"
                "vqshrun.s16 d4,  q3, #6        \n"
                "vqshrun.s16 d25, q9, #6        \n"   // d4-d6: b1g1r1
                "vqshrun.s16 d6, q10, #6        \n"
                "vqshrun.s16 d5, q11, #6        \n"
                "pld        [%[_vu], #128]      \n"
                "vld1.u8    {d2}, [%[_vu]]!     \n"
                "subs       %[_nn], #1          \n"
                "vst3.u8    {d24-d26}, [%[_r0]]!\n"
                "vcgt.u8    d27, d2, %[_v240]   \n"
                "vbsl.u8    d27,  %[_v240], d2  \n"
                "vsub.u8    d2, d27, %[_v128]   \n"
                "vst3.u8    {d4-d6},   [%[_r1]]!\n"
                "bne        0b                  \n"
                "sub        %[_vu], #8          \n"

                : [_nn]"+r"(nn),
                  [_y0]"+r"(yptr0),
                  [_y1]"+r"(yptr1),
                  [_vu]"+r"(vuptr),
                  [_r0]"+r"(rgb0),
                  [_r1]"+r"(rgb1)
                : [_v128]"w"(_v128),
                  [_filt]"w"(_vuvfilter),
                  [_v74]"w"(_v74),
                  [_s1135]"r"(_s1135),
                  [_v240]"w"(_v240)
                : "cc", "memory", "q0", "q1", "q2", "q3","q4","q5","q6","q7","q8", "q9", "q10", "q11", "q12", "q13"
            );
        }
#endif //__aarch64__
        NaiveYUVToBGROrBGRALoop(yptr0, yptr1, vuptr, rgb0, rgb1, remain, true, 3);
        yptr  += 2*w;
        vuptr += remain;
        bgr   += 2*3*w;
    }
#endif
}

void NV21ToBGR(const unsigned char* nv21, unsigned char* bgr, int h, int w) {
#ifndef TNN_USE_NEON
    return NaiveYUVToBGROrBGRA(nv21, bgr, 3, h, w, false);
#else
    const unsigned char* yptr  = nv21;
    const unsigned char* vuptr = nv21 + w * h;

    for (int y = 0; y < h; y += 2) {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* rgb0 = bgr;
        unsigned char* rgb1 = bgr + w * 3;
#if __aarch64__
        int64_t nn = w >> 3;
        int remain = w - (nn << 3);

        int16x8_t _q1135 = vdupq_n_s16(1135);
        int8x8_t _v74    = vdup_n_s8(74);
        int8x8_t _v128   = vdup_n_s8(int8_t(128));
        int8x8_t _v102   = vdup_n_s8(102);
        int8x8_t _v52    = vdup_n_s8(52);
        int8x8_t _v25    = vdup_n_s8(25);
        // use 127 instead of 129 to prevent char overflow, add another 2 in asm
        int8x8_t _v127   = vdup_n_s8(127);
        // saturate uv to 240 to avoid b overflow
        uint8x8_t _v240  = vdup_n_u8(240);

        if (nn > 0) {
            asm volatile(
                "prfm  pldl1strm, [%[_vu], #128]    \n\t"
                "ld1   {v2.8b},   [%[_vu]], #8      \n\t"
                "cmhi  v12.8b, v2.8b, %[_v240].8b   \n\t"
                "bsl   v12.8b, %[_v240].8b, v2.8b   \n\t"
                "sub   v2.8b, v12.8b, %[_v128].8b   \n\t"
                "0:                                 \n\t"
                "prfm  pldl1strm, [%[_y0], #128]    \n\t"
                "ld1   {v0.8b},   [%[_y0]], #8      \n\t"
                "prfm  pldl1strm, [%[_y1], #128]    \n\t"
                "ld1   {v1.8b},   [%[_y1]], #8      \n\t"
                "umull v28.8h, v0.8b,  %[_v74].8b   \n\t"
                "sub   v28.8h, v28.8h, %[_q1135].8h \n\t"   // v28 -> b0
                "orr   v3.8b,  v2.8b,  v2.8b        \n\t"
                "umull v29.8h, v1.8b,  %[_v74].8b   \n\t"
                "sub   v29.8h, v29.8h, %[_q1135].8h \n\t"   // v29 -> b1
                "orr   v9.16b, v28.16b, v28.16b     \n\t"   // v9  -> g0
                "trn1  v30.8b, v2.8b, v3.8b         \n\t"   // u
                "trn2  v31.8b, v2.8b, v3.8b         \n\t"   // v
                "orr   v11.16b, v29.16b, v29.16b    \n\t"   // v11 -> g1
                "sshll v27.8h, v31.8b, #1           \n\t"
                "smlsl v9.8h,  v30.8b, %[_v52].8b   \n\t"
                "orr   v8.16b, v28.16b, v28.16b     \n\t"   // v8  -> r0
                "smlsl v11.8h, v30.8b, %[_v52].8b   \n\t"
                "orr   v10.16b, v29.16b, v29.16b    \n\t"   // v10 -> r1
                "smlal v8.8h,  v30.8b, %[_v102].8b  \n\t"
                "smlal v28.8h, v31.8b, %[_v127].8b  \n\t"
                "smlal v10.8h, v30.8b, %[_v102].8b  \n\t"
                "add   v28.8h, v28.8h, v27.8h       \n\t"
                "smlsl v9.8h,  v31.8b, %[_v25].8b   \n\t"
                "smlal v29.8h, v31.8b, %[_v127].8b  \n\t"
                "smlsl v11.8h, v31.8b, %[_v25].8b   \n\t"
                "add   v29.8h, v29.8h, v27.8h       \n\t"
                "sqshrun v26.8b, v8.8h,  #6         \n\t"   // v24-v26: b0g0r0
                "sqshrun v24.8b, v28.8h, #6         \n\t"
                "sqshrun v6.8b,  v10.8h, #6         \n\t"
                "sqshrun v25.8b, v9.8h,  #6         \n\t"   // v4-v6: b1g1r1
                "sqshrun v4.8b,  v29.8h, #6         \n\t"
                "sqshrun v5.8b,  v11.8h, #6         \n\t"
                "prfm pldl1strm, [%[_vu], #128]     \n\t"
                "ld1 {v2.8b},    [%[_vu]], #8       \n\t"
                "subs %[_nn], %[_nn], #1            \n\t"
                "prfm pstl1strm, [%[_r0]]           \n\t"
                "st3 {v24.8b-v26.8b}, [%[_r0]], #24 \n\t"
                "cmhi  v12.8b, v2.8b, %[_v240].8b   \n\t"
                "bsl   v12.8b, %[_v240].8b, v2.8b   \n\t"
                "sub   v2.8b, v12.8b, %[_v128].8b   \n\t"
                "prfm pstl1strm, [%[_r1]]           \n\t"
                "st3 {v4.8b-v6.8b},   [%[_r1]], #24 \n\t"
                "bne 0b                             \n\t"
                "sub %[_vu], %[_vu], #8             \n\t"

                : [_nn]"+r"(nn),
                  [_y0]"+r"(yptr0),
                  [_y1]"+r"(yptr1),
                  [_vu]"+r"(vuptr),
                  [_r0]"+r"(rgb0),
                  [_r1]"+r"(rgb1)
                : [_v128]"w"(_v128),
                  [_v102]"w"(_v102),
                  [_v52]"w"(_v52),
                  [_v25]"w"(_v25),
                  [_v127]"w"(_v127),
                  [_q1135]"w"(_q1135),
                  [_v74]"w"(_v74),
                  [_v240]"w"(_v240)
                : "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8",
                  "v9", "v10", "v11", "v12", "v24", "v25", "v26","v27", "v28", "v29", "v30", "v31"
            );
        }
#else
        int nn         = w >> 3;
        int remain     = w - (nn << 3);
        short _s1135   = 1135;
        int8x8_t _v74  = vdup_n_s8(74);
        int8x8_t _v128 = vdup_n_s8(int8_t(128));
        // to much input w cause compile error, merge to one
        int8x8_t _vuvfilter = {102, 52, 25, 127, 0, 0, 0, 0};
        // saturate uv to 240 to avoid b overflow
        uint8x8_t _v240     = vdup_n_u8(240);

        if (nn > 0) {
            asm volatile(
                "pld        [%[_vu], #128]      \n"
                "vld1.u8    {d2}, [%[_vu]]!     \n"
                "vcgt.u8    d27, d2, %[_v240]   \n"
                "vbsl.u8    d27,  %[_v240], d2  \n"
                "vsub.u8    d2, d27, %[_v128]   \n"
                "vmov.s8    d10, %[_filt]       \n"
                "vdup.8     d11, d10[1]         \n"   // v52
                "vdup.8     d12, d10[2]         \n"   // v25
                "vdup.8     d13, d10[3]         \n"   // v127
                "vdup.16    q7,  %[_s1135]      \n"   // q1135
                "vdup.8     d10, d10[0]         \n"   // v102
                "0:                             \n"
                "pld        [%[_y0], #128]      \n"
                "vld1.u8    {d0}, [%[_y0]]!     \n"
                "pld        [%[_y1], #128]      \n"
                "vld1.u8    {d1}, [%[_y1]]!     \n"
                "vmull.u8   q2, d0, %[_v74]     \n"
                "vorr       d3, d2, d2          \n"
                "vsub.s16   q2, q2, q7          \n"   // q2  -> b0
                "vmull.u8   q3, d1, %[_v74]     \n"
                "vorr       q9, q2, q2          \n"   // q9  -> g0
                "vsub.s16   q3, q3, q7          \n"   // q3  -> b1
                "vtrn.s8    d2, d3              \n"   // d2 -> u, d3 -> v
                "vorr       q11, q3, q3         \n"   // q11 -> g1
                "vshll.s8   q4, d3, #1          \n"
                "vmlsl.s8   q9, d2, d11         \n"
                "vorr       q8, q2, q2          \n"   // q8  -> r0
                "vmlsl.s8   q11, d2, d11        \n"
                "vorr       q10, q3, q3         \n"   // q10 -> r1
                "vmlal.s8   q8, d2, d10         \n"
                "vmlal.s8   q2, d3, d13         \n"
                "vmlal.s8   q10, d2, d10        \n"
                "vadd.s16   q2, q2, q4          \n"
                "vmlsl.s8   q9, d3, d12         \n"
                "vmlal.s8   q3, d3, d13         \n"
                "vmlsl.s8   q11,d3, d12         \n"
                "vadd.s16   q3, q3, q4          \n"
                "vqshrun.s16 d26, q8, #6        \n"   // d24-d26: b0g0r0
                "vqshrun.s16 d24, q2, #6        \n"
                "vqshrun.s16 d4,  q3, #6        \n"
                "vqshrun.s16 d25, q9, #6        \n"   // d4-d6: b1g1r1
                "vqshrun.s16 d6, q10, #6        \n"
                "vqshrun.s16 d5, q11, #6        \n"
                "pld        [%[_vu], #128]      \n"
                "vld1.u8    {d2}, [%[_vu]]!     \n"
                "subs       %[_nn], #1          \n"
                "vst3.u8    {d24-d26}, [%[_r0]]!\n"
                "vcgt.u8    d27, d2, %[_v240]   \n"
                "vbsl.u8    d27,  %[_v240], d2  \n"
                "vsub.u8    d2, d27, %[_v128]   \n"
                "vst3.u8    {d4-d6},   [%[_r1]]!\n"
                "bne        0b                  \n"
                "sub        %[_vu], #8          \n"

                : [_nn]"+r"(nn),
                  [_y0]"+r"(yptr0),
                  [_y1]"+r"(yptr1),
                  [_vu]"+r"(vuptr),
                  [_r0]"+r"(rgb0),
                  [_r1]"+r"(rgb1)
                : [_v128]"w"(_v128),
                  [_filt]"w"(_vuvfilter),
                  [_v74]"w"(_v74),
                  [_s1135]"r"(_s1135),
                  [_v240]"w"(_v240)
                : "cc", "memory", "q0", "q1", "q2", "q3","q4","q5","q6","q7","q8", "q9", "q10", "q11", "q12", "q13"
            );
        }
#endif //__aarch64__
        NaiveYUVToBGROrBGRALoop(yptr0, yptr1, vuptr, rgb0, rgb1, remain, false, 3);
        yptr  += 2*w;
        vuptr += remain;
        bgr   += 2*3*w;
    }
#endif // TNN_USE_NEON
}

void NV12ToBGRA(const unsigned char* nv12, unsigned char* bgra, int h, int w) {
#ifndef TNN_USE_NEON
    return NaiveYUVToBGROrBGRA(nv12, bgra, 4, h, w, true);
#else
    const unsigned char* yptr  = nv12;
    const unsigned char* vuptr = nv12 + w * h;

    for (int y = 0; y < h; y += 2) {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* rgb0 = bgra;
        unsigned char* rgb1 = bgra + w * 4;
#if __aarch64__
        int64_t nn = w >> 3;
        int remain = w - (nn << 3);

        int16x8_t _q1135 = vdupq_n_s16(1135);
        int8x8_t _v74    = vdup_n_s8(74);
        int8x8_t _v128   = vdup_n_s8(int8_t(128));
        int8x8_t _v255   = vdup_n_s8(int8_t(255));
        int8x8_t _v102   = vdup_n_s8(102);
        int8x8_t _v52    = vdup_n_s8(52);
        int8x8_t _v25    = vdup_n_s8(25);
        // use 127 instead of 129 to prevent char overflow, add another 2 in asm
        int8x8_t _v127   = vdup_n_s8(127);
        // saturate uv to 240 to avoid b overflow
        uint8x8_t _v240  = vdup_n_u8(240);

        if (nn > 0) {
            asm volatile(
                "prfm  pldl1strm, [%[_vu], #128]    \n\t"
                "ld1   {v2.8b},   [%[_vu]], #8      \n\t"
                "cmhi  v12.8b, v2.8b, %[_v240].8b   \n\t"
                "bsl   v12.8b, %[_v240].8b, v2.8b   \n\t"
                "sub   v2.8b, v12.8b, %[_v128].8b   \n\t"
                "mov   v27.8b, %[_v255].8b          \n\t"
                "orr   v7.8b,  v27.8b, v27.8b       \n\t"
                "0:                                 \n\t"
                "prfm  pldl1strm, [%[_y0], #128]    \n\t"
                "ld1   {v0.8b},   [%[_y0]], #8      \n\t"
                "prfm  pldl1strm, [%[_y1], #128]    \n\t"
                "ld1   {v1.8b},   [%[_y1]], #8      \n\t"
                "umull v28.8h, v0.8b,  %[_v74].8b   \n\t"
                "sub   v28.8h, v28.8h, %[_q1135].8h \n\t"   // v28 -> b0
                "orr   v3.8b,  v2.8b,  v2.8b        \n\t"
                "umull v29.8h, v1.8b,  %[_v74].8b   \n\t"
                "sub   v29.8h, v29.8h, %[_q1135].8h \n\t"   // v29 -> b1
                "orr   v9.16b, v28.16b, v28.16b     \n\t"   // v9  -> g0
                "trn1  v31.8b, v2.8b, v3.8b         \n\t"   // u
                "trn2  v30.8b, v2.8b, v3.8b         \n\t"   // v
                "orr   v11.16b, v29.16b, v29.16b    \n\t"   // v11 -> g1
                "sshll v13.8h, v31.8b, #1           \n\t"
                "smlsl v9.8h,  v30.8b, %[_v52].8b   \n\t"
                "orr   v8.16b, v28.16b, v28.16b     \n\t"   // v8  -> r0
                "smlsl v11.8h, v30.8b, %[_v52].8b   \n\t"
                "orr   v10.16b, v29.16b, v29.16b    \n\t"   // v10 -> r1
                "smlal v8.8h,  v30.8b, %[_v102].8b  \n\t"
                "smlal v28.8h, v31.8b, %[_v127].8b  \n\t"
                "smlal v10.8h, v30.8b, %[_v102].8b  \n\t"
                "add   v28.8h, v28.8h, v13.8h       \n\t"
                "smlsl v9.8h,  v31.8b, %[_v25].8b   \n\t"
                "smlal v29.8h, v31.8b, %[_v127].8b  \n\t"
                "smlsl v11.8h, v31.8b, %[_v25].8b   \n\t"
                "add   v29.8h, v29.8h, v13.8h       \n\t"
                "sqshrun v26.8b, v8.8h,  #6         \n\t"   // v24-v27: b0g0r0a0
                "sqshrun v24.8b, v28.8h, #6         \n\t"
                "sqshrun v6.8b,  v10.8h, #6         \n\t"
                "sqshrun v25.8b, v9.8h,  #6         \n\t"   // v4-v7: b1g1r1a1
                "sqshrun v4.8b,  v29.8h, #6         \n\t"
                "sqshrun v5.8b,  v11.8h, #6         \n\t"
                "prfm pldl1strm, [%[_vu], #128]     \n\t"
                "ld1 {v2.8b},    [%[_vu]], #8       \n\t"
                "subs %[_nn], %[_nn], #1            \n\t"
                "prfm pstl1strm, [%[_r0]]           \n\t"
                "st4 {v24.8b-v27.8b}, [%[_r0]], #32 \n\t"
                "cmhi  v12.8b, v2.8b, %[_v240].8b   \n\t"
                "bsl   v12.8b, %[_v240].8b, v2.8b   \n\t"
                "sub   v2.8b, v12.8b, %[_v128].8b   \n\t"
                "prfm pstl1strm, [%[_r1]]           \n\t"
                "st4 {v4.8b-v7.8b},   [%[_r1]], #32 \n\t"
                "bne 0b                             \n\t"
                "sub %[_vu], %[_vu], #8             \n\t"

                : [_nn]"+r"(nn),
                  [_y0]"+r"(yptr0),
                  [_y1]"+r"(yptr1),
                  [_vu]"+r"(vuptr),
                  [_r0]"+r"(rgb0),
                  [_r1]"+r"(rgb1)
                : [_v128]"w"(_v128),
                  [_v102]"w"(_v102),
                  [_v52]"w"(_v52),
                  [_v25]"w"(_v25),
                  [_v127]"w"(_v127),
                  [_q1135]"w"(_q1135),
                  [_v74]"w"(_v74),
                  [_v240]"w"(_v240),
                  [_v255]"w"(_v255)
                : "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                  "v9", "v10", "v11", "v12", "v13", "v24", "v25", "v26","v27", "v28", "v29", "v30", "v31"
            );
        }
#else
        int nn         = w >> 3;
        int remain     = w - (nn << 3);
        short _s1135   = 1135;
        int8x8_t _v74  = vdup_n_s8(74);
        int8x8_t _v128 = vdup_n_s8(int8_t(128));
        // to much input w cause compile error, merge to one
        int8x8_t _vuvfilter = {102, 52, 25, 127, int8_t(255), 0, 0, 0};
        // saturate uv to 240 to avoid b overflow
        uint8x8_t _v240     = vdup_n_u8(240);

        if (nn > 0) {
            asm volatile(
                "pld        [%[_vu], #128]      \n"
                "vld1.u8    {d2}, [%[_vu]]!     \n"
                "vcgt.u8    d9, d2, %[_v240]    \n"
                "vbsl.u8    d9,  %[_v240], d2   \n"
                "vsub.u8    d2, d9, %[_v128]    \n"
                "vmov.s8    d10, %[_filt]       \n"
                "vdup.8     d27, d10[4]         \n"   // v255
                "vdup.8     d11, d10[1]         \n"   // v52
                "vdup.8     d12, d10[2]         \n"   // v25
                "vdup.8     d13, d10[3]         \n"   // v127
                "vdup.16    q7,  %[_s1135]      \n"   // q1135
                "vdup.8     d10, d10[0]         \n"   // v102
                "0:                             \n"
                "pld        [%[_y0], #128]      \n"
                "vld1.u8    {d0}, [%[_y0]]!     \n"
                "pld        [%[_y1], #128]      \n"
                "vld1.u8    {d1}, [%[_y1]]!     \n"
                "vmull.u8   q2, d0, %[_v74]     \n"
                "vorr       d3, d2, d2          \n"
                "vsub.s16   q2, q2, q7          \n"   // q2  -> b0
                "vmull.u8   q3, d1, %[_v74]     \n"
                "vorr       q9, q2, q2          \n"   // q9  -> g0
                "vsub.s16   q3, q3, q7          \n"   // q3  -> b1
                "vtrn.s8    d3, d2              \n"   // d3 -> u, d2 -> v
                "vorr       q11, q3, q3         \n"   // q11 -> g1
                "vshll.s8   q4, d3, #1          \n"
                "vmlsl.s8   q9, d2, d11         \n"
                "vorr       q8, q2, q2          \n"   // q8  -> r0
                "vmlsl.s8   q11, d2, d11        \n"
                "vorr       q10, q3, q3         \n"   // q10 -> r1
                "vmlal.s8   q8, d2, d10         \n"
                "vmlal.s8   q2, d3, d13         \n"
                "vmlal.s8   q10, d2, d10        \n"
                "vadd.s16   q2, q2, q4          \n"
                "vmlsl.s8   q9, d3, d12         \n"
                "vmlal.s8   q3, d3, d13         \n"
                "vmlsl.s8   q11,d3, d12         \n"
                "vadd.s16   q3, q3, q4          \n"
                "vqshrun.s16 d26, q8, #6        \n"   // d24-d27: b0g0r0a0
                "vqshrun.s16 d24, q2, #6        \n"
                "vqshrun.s16 d3,  q3, #6        \n"
                "vqshrun.s16 d25, q9, #6        \n"   // d3-d6: b1g1r1a1
                "vqshrun.s16 d5, q10, #6        \n"
                "vqshrun.s16 d4, q11, #6        \n"
                "pld        [%[_vu], #128]      \n"
                "vld1.u8    {d2}, [%[_vu]]!     \n"
                "vorr       d6, d27, d27        \n"
                "subs       %[_nn], #1          \n"
                "vst4.u8    {d24-d27}, [%[_r0]]!\n"
                "vcgt.u8    d9, d2, %[_v240]    \n"
                "vbsl.u8    d9,  %[_v240], d2   \n"
                "vsub.u8    d2, d9, %[_v128]    \n"
                "vst4.u8    {d3-d6},   [%[_r1]]!\n"
                "bne        0b                  \n"
                "sub        %[_vu], #8          \n"

                : [_nn]"+r"(nn),
                  [_y0]"+r"(yptr0),
                  [_y1]"+r"(yptr1),
                  [_vu]"+r"(vuptr),
                  [_r0]"+r"(rgb0),
                  [_r1]"+r"(rgb1)
                : [_v128]"w"(_v128),
                  [_filt]"w"(_vuvfilter),
                  [_v74]"w"(_v74),
                  [_s1135]"r"(_s1135),
                  [_v240]"w"(_v240)
                : "cc", "memory", "q0", "q1", "q2", "q3","q4","q5","q6","q7","q8", "q9", "q10", "q11", "q12", "q13"
            );
        }
#endif //__aarch64__
        NaiveYUVToBGROrBGRALoop(yptr0, yptr1, vuptr, rgb0, rgb1, remain, true, 4);
        yptr  += 2*w;
        vuptr += remain;
        bgra  += 2*4*w;
    }
#endif // TNN_USE_NEON
}

void NV21ToBGRA(const unsigned char* nv21, unsigned char* bgra, int h, int w) {
#ifndef TNN_USE_NEON
    return NaiveYUVToBGROrBGRA(nv21, bgra, 4, h, w, false);
#else
    const unsigned char* yptr  = nv21;
    const unsigned char* vuptr = nv21 + w * h;

    for (int y = 0; y < h; y += 2) {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* rgb0 = bgra;
        unsigned char* rgb1 = bgra + w * 4;
#if __aarch64__
        int64_t nn = w >> 3;
        int remain = w - (nn << 3);

        int16x8_t _q1135 = vdupq_n_s16(1135);
        int8x8_t _v74    = vdup_n_s8(74);
        int8x8_t _v128   = vdup_n_s8(int8_t(128));
        int8x8_t _v255   = vdup_n_s8(int8_t(255));
        int8x8_t _v102   = vdup_n_s8(102);
        int8x8_t _v52    = vdup_n_s8(52);
        int8x8_t _v25    = vdup_n_s8(25);
        // use 127 instead of 129 to prevent char overflow, add another 2 in asm
        int8x8_t _v127   = vdup_n_s8(127);
        // saturate uv to 240 to avoid b overflow
        uint8x8_t _v240  = vdup_n_u8(240);

        if (nn > 0) {
            asm volatile(
                "prfm  pldl1strm, [%[_vu], #128]    \n\t"
                "ld1   {v2.8b},   [%[_vu]], #8      \n\t"
                "cmhi  v12.8b, v2.8b, %[_v240].8b   \n\t"
                "bsl   v12.8b, %[_v240].8b, v2.8b   \n\t"
                "sub   v2.8b, v12.8b, %[_v128].8b   \n\t"
                "mov   v27.8b, %[_v255].8b          \n\t"
                "orr   v7.8b,  v27.8b, v27.8b       \n\t"
                "0:                                 \n\t"
                "prfm  pldl1strm, [%[_y0], #128]    \n\t"
                "ld1   {v0.8b},   [%[_y0]], #8      \n\t"
                "prfm  pldl1strm, [%[_y1], #128]    \n\t"
                "ld1   {v1.8b},   [%[_y1]], #8      \n\t"
                "umull v28.8h, v0.8b,  %[_v74].8b   \n\t"
                "sub   v28.8h, v28.8h, %[_q1135].8h \n\t"   // v28 -> b0
                "orr   v3.8b,  v2.8b,  v2.8b        \n\t"
                "umull v29.8h, v1.8b,  %[_v74].8b   \n\t"
                "sub   v29.8h, v29.8h, %[_q1135].8h \n\t"   // v29 -> b1
                "orr   v9.16b, v28.16b, v28.16b     \n\t"   // v9  -> g0
                "trn1  v30.8b, v2.8b, v3.8b         \n\t"   // u
                "trn2  v31.8b, v2.8b, v3.8b         \n\t"   // v
                "orr   v11.16b, v29.16b, v29.16b    \n\t"   // v11 -> g1
                "sshll v13.8h, v31.8b, #1           \n\t"
                "smlsl v9.8h,  v30.8b, %[_v52].8b   \n\t"
                "orr   v8.16b, v28.16b, v28.16b     \n\t"   // v8  -> r0
                "smlsl v11.8h, v30.8b, %[_v52].8b   \n\t"
                "orr   v10.16b, v29.16b, v29.16b    \n\t"   // v10 -> r1
                "smlal v8.8h,  v30.8b, %[_v102].8b  \n\t"
                "smlal v28.8h, v31.8b, %[_v127].8b  \n\t"
                "smlal v10.8h, v30.8b, %[_v102].8b  \n\t"
                "add   v28.8h, v28.8h, v13.8h       \n\t"
                "smlsl v9.8h,  v31.8b, %[_v25].8b   \n\t"
                "smlal v29.8h, v31.8b, %[_v127].8b  \n\t"
                "smlsl v11.8h, v31.8b, %[_v25].8b   \n\t"
                "add   v29.8h, v29.8h, v13.8h       \n\t"
                "sqshrun v26.8b, v8.8h,  #6         \n\t"   // v24-v27: b0g0r0a0
                "sqshrun v24.8b, v28.8h, #6         \n\t"
                "sqshrun v6.8b,  v10.8h, #6         \n\t"
                "sqshrun v25.8b, v9.8h,  #6         \n\t"   // v4-v7: b1g1r1a1
                "sqshrun v4.8b,  v29.8h, #6         \n\t"
                "sqshrun v5.8b,  v11.8h, #6         \n\t"
                "prfm pldl1strm, [%[_vu], #128]     \n\t"
                "ld1 {v2.8b},    [%[_vu]], #8       \n\t"
                "subs %[_nn], %[_nn], #1            \n\t"
                "prfm pstl1strm, [%[_r0]]           \n\t"
                "st4 {v24.8b-v27.8b}, [%[_r0]], #32 \n\t"
                "cmhi  v12.8b, v2.8b, %[_v240].8b   \n\t"
                "bsl   v12.8b, %[_v240].8b, v2.8b   \n\t"
                "sub   v2.8b, v12.8b, %[_v128].8b   \n\t"
                "prfm pstl1strm, [%[_r1]]           \n\t"
                "st4 {v4.8b-v7.8b},   [%[_r1]], #32 \n\t"
                "bne 0b                             \n\t"
                "sub %[_vu], %[_vu], #8             \n\t"

                : [_nn]"+r"(nn),
                  [_y0]"+r"(yptr0),
                  [_y1]"+r"(yptr1),
                  [_vu]"+r"(vuptr),
                  [_r0]"+r"(rgb0),
                  [_r1]"+r"(rgb1)
                : [_v128]"w"(_v128),
                  [_v102]"w"(_v102),
                  [_v52]"w"(_v52),
                  [_v25]"w"(_v25),
                  [_v127]"w"(_v127),
                  [_q1135]"w"(_q1135),
                  [_v74]"w"(_v74),
                  [_v240]"w"(_v240),
                  [_v255]"w"(_v255)
                : "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                  "v9", "v10", "v11", "v12", "v13", "v24", "v25", "v26","v27", "v28", "v29", "v30", "v31"
            );
        }
#else
        int nn         = w >> 3;
        int remain     = w - (nn << 3);
        short _s1135   = 1135;
        int8x8_t _v74  = vdup_n_s8(74);
        int8x8_t _v128 = vdup_n_s8(int8_t(128));
        // to much input w cause compile error, merge to one
        int8x8_t _vuvfilter = {102, 52, 25, 127, int8_t(255), 0, 0, 0};
        // saturate uv to 240 to avoid b overflow
        uint8x8_t _v240     = vdup_n_u8(240);

        if (nn > 0) {
            asm volatile(
                "pld        [%[_vu], #128]      \n"
                "vld1.u8    {d2}, [%[_vu]]!     \n"
                "vcgt.u8    d9, d2, %[_v240]    \n"
                "vbsl.u8    d9,  %[_v240], d2   \n"
                "vsub.u8    d2, d9, %[_v128]    \n"
                "vmov.s8    d10, %[_filt]       \n"
                "vdup.8     d27, d10[4]         \n"   // v255
                "vdup.8     d11, d10[1]         \n"   // v52
                "vdup.8     d12, d10[2]         \n"   // v25
                "vdup.8     d13, d10[3]         \n"   // v127
                "vdup.16    q7,  %[_s1135]      \n"   // q1135
                "vdup.8     d10, d10[0]         \n"   // v102
                "0:                             \n"
                "pld        [%[_y0], #128]      \n"
                "vld1.u8    {d0}, [%[_y0]]!     \n"
                "pld        [%[_y1], #128]      \n"
                "vld1.u8    {d1}, [%[_y1]]!     \n"
                "vmull.u8   q2, d0, %[_v74]     \n"
                "vorr       d3, d2, d2          \n"
                "vsub.s16   q2, q2, q7          \n"   // q2  -> b0
                "vmull.u8   q3, d1, %[_v74]     \n"
                "vorr       q9, q2, q2          \n"   // q9  -> g0
                "vsub.s16   q3, q3, q7          \n"   // q3  -> b1
                "vtrn.s8    d2, d3              \n"   // d2 -> u, d3 -> v
                "vorr       q11, q3, q3         \n"   // q11 -> g1
                "vshll.s8   q4, d3, #1          \n"
                "vmlsl.s8   q9, d2, d11         \n"
                "vorr       q8, q2, q2          \n"   // q8  -> r0
                "vmlsl.s8   q11, d2, d11        \n"
                "vorr       q10, q3, q3         \n"   // q10 -> r1
                "vmlal.s8   q8, d2, d10         \n"
                "vmlal.s8   q2, d3, d13         \n"
                "vmlal.s8   q10, d2, d10        \n"
                "vadd.s16   q2, q2, q4          \n"
                "vmlsl.s8   q9, d3, d12         \n"
                "vmlal.s8   q3, d3, d13         \n"
                "vmlsl.s8   q11,d3, d12         \n"
                "vadd.s16   q3, q3, q4          \n"
                "vqshrun.s16 d26, q8, #6        \n"   // d24-d27: b0g0r0a0
                "vqshrun.s16 d24, q2, #6        \n"
                "vqshrun.s16 d3,  q3, #6        \n"
                "vqshrun.s16 d25, q9, #6        \n"   // d3-d6: b1g1r1a1
                "vqshrun.s16 d5, q10, #6        \n"
                "vqshrun.s16 d4, q11, #6        \n"
                "pld        [%[_vu], #128]      \n"
                "vld1.u8    {d2}, [%[_vu]]!     \n"
                "vorr       d6, d27, d27        \n"
                "subs       %[_nn], #1          \n"
                "vst4.u8    {d24-d27}, [%[_r0]]!\n"
                "vcgt.u8    d9, d2, %[_v240]    \n"
                "vbsl.u8    d9,  %[_v240], d2   \n"
                "vsub.u8    d2, d9, %[_v128]    \n"
                "vst4.u8    {d3-d6},   [%[_r1]]!\n"
                "bne        0b                  \n"
                "sub        %[_vu], #8          \n"

                : [_nn]"+r"(nn),
                  [_y0]"+r"(yptr0),
                  [_y1]"+r"(yptr1),
                  [_vu]"+r"(vuptr),
                  [_r0]"+r"(rgb0),
                  [_r1]"+r"(rgb1)
                : [_v128]"w"(_v128),
                  [_filt]"w"(_vuvfilter),
                  [_v74]"w"(_v74),
                  [_s1135]"r"(_s1135),
                  [_v240]"w"(_v240)
                : "cc", "memory", "q0", "q1", "q2", "q3","q4","q5","q6","q7","q8", "q9", "q10", "q11", "q12", "q13"
            );
        }
#endif //__aarch64__
        NaiveYUVToBGROrBGRALoop(yptr0, yptr1, vuptr, rgb0, rgb1, remain, false, 4);
        yptr  += 2*w;
        vuptr += remain;
        bgra  += 2*4*w;
    }
#endif // TNN_USE_NEON
}

#ifdef TNN_USE_NEON

#define CVTGRAYIMPL(n)                                                  \
    uint8x8x##n##_t _Src;                                               \
    _Src  = vld##n##_u8(Sp);                                            \
    _Bh   = vmovl_u8(_Src.val[0]);                                      \
    _Gh   = vmovl_u8(_Src.val[1]);                                      \
    _Rh   = vmovl_u8(_Src.val[2]);                                      \
    _Bval = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_Bh)));                \
    _Gval = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_Gh)));                \
    _Rval = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_Rh)));                \
    _acc  = _Bval * _coeff_b;                                           \
    _acc  = _acc + _Gval * _coeff_g;                                    \
    _acc  = _acc + _Rval * _coeff_r;                                    \
    _acc0 = vmovn_u32(vcvtq_u32_f32(_acc.value));                       \
    _Bval = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_Bh)));               \
    _Gval = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_Gh)));               \
    _Rval = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_Rh)));               \
    _acc  = _Bval * _coeff_b;                                           \
    _acc  = _acc + _Gval * _coeff_g;                                    \
    _acc  = _acc + _Rval * _coeff_r;                                    \
    _acc1 = vmovn_u32(vcvtq_u32_f32(_acc.value));                       \
    vst1_u8(Dp, vmovn_u16(vcombine_u16(_acc0, _acc1)));                 \

#endif  // TNN_USE_NEON

template <int channel>
void BGROrBGRAToGray(const unsigned char* bgr, unsigned char* gray, int h, int w) {
#ifndef TNN_USE_NEON
    NaiveBGROrBGRAToGray(bgr, gray, h, w, channel);
#else
    int offset = 0;
    int plane  = h * w;

    const unsigned char* Sp = bgr;
    unsigned char* Dp       = gray;
    uint16x8_t _Bh, _Gh, _Rh;
    Float4 _Bval, _Gval, _Rval, _acc;
    Float4 _coeff_b(0.114);
    Float4 _coeff_g(0.587);
    Float4 _coeff_r(0.299);
    uint16x4_t _acc0, _acc1;
    for (; offset < plane>>3<<3; offset += 8) {
        if (channel == 3) {
            CVTGRAYIMPL(3);
        } else {
            CVTGRAYIMPL(4);
        }
        Sp   += 8 * channel;
        Dp   += 8;
    }
    if (plane % 8) {
        offset -= 8;
    }

    for (; offset < plane; ++offset) {
        unsigned b = bgr[offset * channel + 0];
        unsigned g = bgr[offset * channel + 1];
        unsigned r = bgr[offset * channel + 2];
        float gray_color = 0.114 * b + 0.587 * g + 0.299 * r;
        gray[offset] = gray_color;
    }
#endif // TNN_USE_NEON
}

void BGRToGray(const unsigned char* bgr, unsigned char* gray, int h, int w) {
    BGROrBGRAToGray<3>(bgr, gray, h, w);
}

void BGRAToGray(const unsigned char* bgra, unsigned char* gray, int h, int w) {
    BGROrBGRAToGray<4>(bgra, gray, h, w);
}

#ifdef TNN_USE_NEON

#undef CVTGRAYIMPL

#endif  // TNN_USE_NEON

}  // namespace TNN_NS
