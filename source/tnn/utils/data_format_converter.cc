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

#include "tnn/utils/data_format_converter.h"

#include "tnn/core/macro.h"

namespace TNN_NS {

template <class T>
static Status ConvertWeightsFromGOIHWToGOIHW16(T *src, T *dst, int group, int input_channel, int output_channel,
                                               int height, int width, bool tanspose = false) {
    const int goc       = output_channel / group;
    const int gic       = input_channel / group;
    const int goc_4     = (goc + 3) / 4;
    const int gic_4     = (gic + 3) / 4;
    const int src_count = group * goc * gic * height * width;

    int src_index = 0;

    if (tanspose) {
        for (int g = 0; g < group; g++) {
            auto g_dst = dst + g * goc_4 * gic_4 * height * width * 16;  // g
#pragma clang loop vectorize(enable)
            for (int i = 0; i < gic; i++) {
                auto zi = i / 4, ri = i % 4;
                auto i_dst = g_dst + zi * height * width * 16 + ri;
#pragma clang loop vectorize(enable)
                for (int o = 0; o < goc; o++) {
                    auto zo = o / 4, ro = o % 4;
                    auto o_dst = i_dst + zo * gic_4 * height * width * 16 + ro * 4;
#pragma clang loop vectorize(enable)
                    for (int h = 0; h < height; h++) {
#pragma clang loop vectorize(enable) unroll(enable)
                        for (int w = 0; w < width; w++) {
                            // to   [g][o/4][i/4][h][w][16]
                            // from [g][i][o][h][w]
                            if (src_index < src_count) {
                                o_dst[(h * width + w) * 16] = src[src_index++];
                            } else {
                                o_dst[(h * width + w) * 16] = 0;
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (int g = 0; g < group; g++) {
            auto g_dst = dst + g * goc_4 * gic_4 * height * width * 16;  // g
#pragma clang loop vectorize(enable)
            for (int o = 0; o < goc; o++) {
                auto zo = o / 4, ro = o % 4;
                auto o_dst = g_dst + zo * gic_4 * height * width * 16 + ro * 4;  // o/4 x 4
#pragma clang loop vectorize(enable)
                for (int i = 0; i < gic; i++) {
                    auto zi = i / 4, ri = i % 4;
                    auto i_dst = o_dst + zi * height * width * 16 + ri;  // i/4 x 4
#pragma clang loop vectorize(enable)
                    for (int h = 0; h < height; h++) {
#pragma clang loop vectorize(enable) unroll(enable)
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
    }

    return TNN_OK;
};

template <class T>
static Status ConvertFromNCHWToNCHW4(T *src, T *dst, int num, int channel, int height, int width,
                                     bool transpose = false) {
    int round_channel = ROUND_UP(channel, 4);
    if (transpose) {
        for (int n = 0; n < num; n++) {
            auto n_dst = dst + n * round_channel * height * width;
            auto n_src = src + n * channel * height * width;
            for (int h = 0; h < height; h++) {
                auto h_dst = n_dst + h * width * 4;
                auto h_src = n_src + h * channel * width;
#pragma clang loop vectorize(enable)
                for (int c = 0; c < round_channel; c++) {
                    auto z = c / 4, r = c % 4;
                    auto z_dst = h_dst + z * height * width * 4 + r;
#pragma clang loop vectorize(enable) unroll(enable)
                    for (int w = 0; w < width; w++) {
                        // to   [c/4][h][w][4]
                        // from [h][c][w]
                        // dst[(z * height  * width + h * width + w) * 4 + r] =
                        // src[ h * channel * width + c * width + w];
                        if (c < channel)
                            z_dst[w * 4] = h_src[c * width + w];
                        else
                            z_dst[w * 4] = 0;
                    }
                }
            }
        }
    } else {
        for (int n = 0; n < num; n++) {
            auto n_dst = dst + n * round_channel * height * width;
            auto n_src = src + n * channel * height * width;
            for (int c = 0; c < round_channel; c++) {
                auto z = c / 4, r = c % 4;
                auto z_dst = n_dst + z * height * width * 4 + r;
                auto z_src = n_src + c * height * width;
#pragma clang loop vectorize(enable)
                for (int h = 0; h < height; h++) {
#pragma clang loop vectorize(enable) unroll(enable)
                    for (int w = 0; w < width; w++) {
                        // to   [c/4][h][w][4]
                        // from [c][h][w]
                        // dst[(z * height * width + h * width + w) * 4 + r] =
                        // src[ c * height * width + h * width + w];
                        if (c < channel)
                            z_dst[(h * width + w) * 4] = z_src[h * width + w];
                        else
                            z_dst[(h * width + w) * 4] = 0;
                    }
                }
            }
        }
    }
    return TNN_OK;
};

template <class T>
static Status ConvertFromNCHWToNHWC4(T *src, T *dst, int num, int channel, int hw) {
    int round_channel = ROUND_UP(channel, 4);
    for (int n = 0; n < num; n++) {
        auto n_dst = dst + n * round_channel * hw;
        auto n_src = src + n * channel * hw;
        for (int z = 0; z < hw; z++) {
            auto z_dst = n_dst + z * round_channel;
            auto z_src = n_src + z;
            for (int c = 0; c < round_channel; c++) {
                // to   [c][hw]
                // from [hw][c4]
                if (c < channel)
                    z_dst[c] = z_src[c * hw];
                else
                    z_dst[c] = 0;
            }
        }
    }
    return TNN_OK;
};

template <class T>
static Status ConvertFromNCHW4ToNCHW(T *src, T *dst, int num, int channel, int height, int width) {
    int round_channel = ROUND_UP(channel, 4);
    for (int n = 0; n < num; n++) {
        auto n_src = src + n * round_channel * height * width;
        auto n_dst = dst + n * channel * height * width;
        for (int c = 0; c < channel; c++) {
            auto z = c / 4, r = c % 4;
            auto z_src = n_src + z * height * width * 4 + r;
            auto z_dst = n_dst + c * height * width;
#pragma clang loop vectorize(enable)
            for (int h = 0; h < height; h++) {
#pragma clang loop vectorize(enable) unroll(enable)
                for (int w = 0; w < width; w++) {
                    // to [c][h][w]
                    // from   [c/4][h][w][4]
                    z_dst[h * width + w] = z_src[(h * width + w) * 4];
                }
            }
        }
    }
    return TNN_OK;
};

template <class T>
static Status ConvertFromNHWC4ToNCHW(T *src, T *dst, int num, int channel, int hw) {
    int round_channel = ROUND_UP(channel, 4);
    for (int n = 0; n < num; n++) {
        auto n_src = src + n * round_channel * hw;
        auto n_dst = dst + n * channel * hw;
        for (int c = 0; c < channel; c++) {
            auto z_src = n_src + c;
            auto z_dst = n_dst + c * hw;
#pragma clang loop vectorize(enable) unroll(enable)
            for (int z = 0; z < hw; z++) {
                // to [c][hw]
                // from   [c/4][hw][4]
                z_dst[z] = z_src[z * round_channel];
            }
        }
    }
    return TNN_OK;
};

Status DataFormatConverter::ConvertFromGOIHWToGOIHW16Float(float *src, float *dst, int group, int input_channel,
                                                           int output_channel, int height, int width, bool tanspose) {
    return ConvertWeightsFromGOIHWToGOIHW16<float>(src, dst, group, input_channel, output_channel, height, width,
                                                   tanspose);
}
Status DataFormatConverter::ConvertFromGOIHWToGOIHW16Half(short *src, short *dst, int group, int input_channel,
                                                          int output_channel, int height, int width, bool tanspose) {
    return ConvertWeightsFromGOIHWToGOIHW16<short>(src, dst, group, input_channel, output_channel, height, width,
                                                   tanspose);
}
Status DataFormatConverter::ConvertFromGOIHWToGOIHW16Int8(int8_t *src, int8_t *dst, int group, int input_channel,
                                                          int output_channel, int height, int width, bool tanspose) {
    return ConvertWeightsFromGOIHWToGOIHW16<int8_t>(src, dst, group, input_channel, output_channel, height, width,
                                                    tanspose);
}

Status DataFormatConverter::ConvertFromInt8ToFloatNCHW4(int8_t *src, float *dst, float *scale, int scale_len, int num,
                                                        int channel, int height, int width) {
    LOGD("scale %g, %d\n", scale[0], scale_len);
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channel; c++) {
            auto z = c / 4, r = c % 4;
            int offset     = n * ROUND_UP(channel, 4) * height * width + z * 4 * height * width + r;
            auto z_src     = src + offset;
            auto z_dst     = dst + offset;
            auto scale_idx = scale_len == 1 ? 0 : c;
#pragma clang loop vectorize(enable)
            for (int hw = 0; hw < height * width; hw++) {
                z_dst[hw * 4] = static_cast<float>(z_src[hw * 4]) * scale[scale_idx];
            }
        }
    }

    return 0;
}

Status DataFormatConverter::ConvertFromInt8ToFloatNHWC4(int8_t *src, float *dst, float *scale, int scale_len, int num,
                                                        int channel, int height, int width) {
    LOGD("scale %g, %d\n", scale[0], scale_len);
    int c_r4 = ROUND_UP(channel, 4);
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channel; c++) {
            auto z = c / 4, r = c % 4;
            int dst_offset = n * c_r4 * height * width + z * 4 * height * width + r;
            int src_offset = n * c_r4 * height * width + c;
            auto z_src     = src + src_offset;
            auto z_dst     = dst + dst_offset;
            auto scale_idx = scale_len == 1 ? 0 : c;
#pragma clang loop vectorize(enable)
            for (int hw = 0; hw < height * width; hw++) {
                z_dst[hw * 4] = static_cast<float>(z_src[hw * c_r4]) * scale[scale_idx];
            }
        }
    }

    return 0;
}

Status DataFormatConverter::ConvertFromInt8ToFloatNCHW(int8_t *src, float *dst, float *scale, int scale_len, int num,
                                                       int channel, int height, int width) {
    LOGD("scale %g, %d\n", scale[0], scale_len);

    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channel; c++) {
            int offset     = n * channel * height * width + c * height * width;
            auto z_src     = src + offset;
            auto z_dst     = dst + offset;
            auto scale_idx = scale_len == 1 ? 0 : c;
#pragma clang loop vectorize(enable)
            for (int hw = 0; hw < height * width; hw++) {
                z_dst[hw] = static_cast<float>(z_src[hw]) * scale[scale_idx];
            }
        }
    }

    return 0;
}

Status DataFormatConverter::ConvertFromNCHWToNCHW4Float(float *src, float *dst, int num, int channel, int height,
                                                        int width, bool transpose) {
    return ConvertFromNCHWToNCHW4<float>(src, dst, num, channel, height, width, transpose);
}
Status DataFormatConverter::ConvertFromNCHWToNCHW4Half(short *src, short *dst, int num, int channel, int height,
                                                       int width, bool transpose) {
    return ConvertFromNCHWToNCHW4<short>(src, dst, num, channel, height, width, transpose);
}

Status DataFormatConverter::ConvertFromNCHWToNHWC4Int8(int8_t *src, int8_t *dst, int num, int channel, int hw) {
    return ConvertFromNCHWToNHWC4<int8_t>(src, dst, num, channel, hw);
}
Status DataFormatConverter::ConvertFromNCHW4ToNCHWFloat(float *src, float *dst, int num, int channel, int height,
                                                        int width) {
    return ConvertFromNCHW4ToNCHW<float>(src, dst, num, channel, height, width);
}
Status DataFormatConverter::ConvertFromNCHW4ToNCHWHalf(short *src, short *dst, int num, int channel, int height,
                                                       int width) {
    return ConvertFromNCHW4ToNCHW<short>(src, dst, num, channel, height, width);
}
Status DataFormatConverter::ConvertFromNHWC4ToNCHWInt8(int8_t *src, int8_t *dst, int num, int channel, int hw) {
    return ConvertFromNHWC4ToNCHW<int8_t>(src, dst, num, channel, hw);
}

}  // namespace TNN_NS
