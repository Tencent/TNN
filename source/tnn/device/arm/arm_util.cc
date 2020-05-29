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

int UnpackAndDequant(float *dst, const int8_t *src, size_t hw, size_t channel, float *scale) {
    int cur_hw;
    int c;
    int idx  = 0;
    int c_r4 = ROUND_UP(channel, 4);
    for (c = 0; c < channel; ++c) {
        auto *src_c = src + c;
        for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
            dst[idx++] = src_c[c_r4 * cur_hw] * scale[c];
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

// to   [g][o/4][i/4][h][w][16]
// from [g][o][i][h][w]
template <typename T>
int ConvertWeightsFromGOIHWToGOIHW16(T *src, T *dst, int group, int input_channel, int output_channel, int height,
                                     int width) {
    const int goc       = output_channel / group;
    const int gic       = input_channel / group;
    const int goc_4     = (goc + 3) / 4;
    const int gic_4     = (gic + 3) / 4;
    const int src_count = group * goc * gic * height * width;

    int src_index = 0;

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
    const int goc       = output_channel / group;
    const int gic       = input_channel / group;
    const int goc_4     = (goc + 3) / 4;
    const int gic_4     = (gic + 3) / 4;
    const int src_count = group * goc * gic * height * width;

    int src_index = 0;

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

}  // namespace TNN_NS
