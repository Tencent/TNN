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

#ifndef TNN_SOURCE_TNN_UTILS_DATA_FORMAT_CONVERTER_H_
#define TNN_SOURCE_TNN_UTILS_DATA_FORMAT_CONVERTER_H_

#include <cstdint>

#include "tnn/core/status.h"

namespace TNN_NS {

class DataFormatConverter {
public:
    // @brief convert weights from [g][o][i][h][w] to [g][o/4][i/4][h][w][16]
    // @param data_tyep data type info
    static Status ConvertFromGOIHWToGOIHW16Float(float *src, float *dst, int group, int input_channel,
                                                 int output_channel, int height, int width, bool tanspose = false);
    static Status ConvertFromGOIHWToGOIHW16Half(short *src, short *dst, int group, int input_channel,
                                                int output_channel, int height, int width, bool tanspose = false);
    static Status ConvertFromGOIHWToGOIHW16Int8(int8_t *src, int8_t *dst, int group, int input_channel,
                                                int output_channel, int height, int width, bool tanspose = false);

    // @brief convert weights from [n][c][h][w] to [n][c/4][h][w][4]
    // @param data_tyep data type info
    static Status ConvertFromNCHWToNCHW4Float(float *src, float *dst, int num, int channel, int height, int width);
    static Status ConvertFromNCHWToNCHW4Half(short *src, short *dst, int num, int channel, int height, int width);
    static Status ConvertFromNCHWToNHWC4Int8(int8_t *src, int8_t *dst, int num, int channel, int height, int width);

    static Status ConvertFromNCHW4ToNCHWFloat(float *src, float *dst, int num, int channel, int height, int width);
    static Status ConvertFromNCHW4ToNCHWHalf(short *src, short *dst, int num, int channel, int height, int width);
    static Status ConvertFromNHWC4ToNCHWInt8(int8_t *src, int8_t *dst, int num, int channel, int height, int width);

    static Status ConvertFromInt8ToFloatNCHW4(int8_t *src, float *dst, float *scale, int scale_len, int num,
                                              int channel, int height, int width);
    static Status ConvertFromInt8ToFloatNCHW(int8_t *src, float *dst, float *scale, int scale_len, int num, int channel,
                                             int height, int width);

    static Status ConvertFromInt8ToFloatNHWC4(int8_t *src, float *dst, float *scale, int scale_len, int num,
                                              int channel, int height, int width);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_UTILS_DATA_FORMAT_CONVERTER_H_
