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
#include <cstring>

#include "tnn/core/blob.h"
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

    enum CVT_DIR { NHWC2NCHW, NCHW2NHWC };

    template <class T>
    static Status ConvertBetweenNHWCAndNCHW(T *data, T *buffer, int num, int channel, int height, int width,
                                            CVT_DIR dir) {
        bool alloc_mem = false;
        if (buffer == nullptr) {
            alloc_mem = true;
            buffer    = new T[num * channel * height * width]();
        }

        auto ptr_nchw = buffer;
        auto ptr_nhwc = data;
        if (NCHW2NHWC == dir)
            std::swap(ptr_nchw, ptr_nhwc);

        for (int n = 0; n < num; ++n) {
            for (int c = 0; c < channel; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        std::swap(ptr_nchw[n * channel * height * width + c * height * width + h * width + w],
                                  ptr_nhwc[n * height * width * channel + h * width * channel + w * channel + c]);
                    }
                }
            }
        }
        if (alloc_mem) {
            memcpy(data, buffer, num * channel * height * width * sizeof(T));
            delete[] buffer;
        }
        return TNN_OK;
    }

    template <class T>
    static Status ConvertFromNCHWToNHWC(Blob *src, Blob *dst) {
        ASSERT(src != nullptr);
        const int num     = src->GetBlobDesc().dims[0];
        const int channel = src->GetBlobDesc().dims[1];
        const int height  = src->GetBlobDesc().dims[2];
        const int width   = src->GetBlobDesc().dims[3];
        T *src_data_ptr   = (T *)src->GetHandle().base;
        T *dst_data_ptr = dst == nullptr ? nullptr : (T *)dst->GetHandle().base;

        auto status = ConvertBetweenNHWCAndNCHW<T>(src_data_ptr, dst_data_ptr, num, channel, height, width, NCHW2NHWC);
        return status;
    }

    template <class T>
    static Status ConvertFromNHWCToNCHW(Blob *src, Blob *dst) {
        ASSERT(src != nullptr);
        const int num     = src->GetBlobDesc().dims[0];
        const int height  = src->GetBlobDesc().dims[2];
        const int width   = src->GetBlobDesc().dims[3];
        const int channel = src->GetBlobDesc().dims[1];
        T *src_data_ptr   = (T *)src->GetHandle().base;
        T *dst_data_ptr   = dst == nullptr ? nullptr : (T *)dst->GetHandle().base;

        auto status = ConvertBetweenNHWCAndNCHW<T>(src_data_ptr, dst_data_ptr, num, channel, height, width, NHWC2NCHW);
        return status;
    }
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_UTILS_DATA_FORMAT_CONVERTER_H_
