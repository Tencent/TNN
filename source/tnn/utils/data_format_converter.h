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
#include "tnn/utils/dims_utils.h"

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
    static Status ConvertFromNCHWToNCHW4Float(float *src, float *dst, int num, int channel, int height, int width, bool transpose = false);
    static Status ConvertFromNCHWToNCHW4Half(short *src, short *dst, int num, int channel, int height, int width, bool transpose = false);
    static Status ConvertFromNCHWToNHWC4Int8(int8_t *src, int8_t *dst, int num, int channel, int hw);

    static Status ConvertFromNCHW4ToNCHWFloat(float *src, float *dst, int num, int channel, int height, int width);
    static Status ConvertFromNCHW4ToNCHWHalf(short *src, short *dst, int num, int channel, int height, int width);
    static Status ConvertFromNHWC4ToNCHWInt8(int8_t *src, int8_t *dst, int num, int channel, int hw);

    static Status ConvertFromInt8ToFloatNCHW4(int8_t *src, float *dst, float *scale, int scale_len, int num,
                                              int channel, int height, int width);
    static Status ConvertFromInt8ToFloatNCHW(int8_t *src, float *dst, float *scale, int scale_len, int num, int channel,
                                             int height, int width);

    static Status ConvertFromInt8ToFloatNHWC4(int8_t *src, float *dst, float *scale, int scale_len, int num,
                                              int channel, int height, int width);

    enum CVT_DIR { NHWC2NCHW, NCHW2NHWC };

    template <class T>
    static Status ConvertBetweenNHWCAndNCHW(T *src, T *dst, int num, int channel, int height, int width, CVT_DIR dir) {
        ASSERT(dir == NHWC2NCHW || dir == NCHW2NHWC);
        bool alloc_mem = false;
        if (dst == nullptr) {
            alloc_mem = true;
            dst       = new T[num * channel * height * width]();
        }

        for (int n = 0; n < num; ++n) {
            for (int c = 0; c < channel; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        // n * channel * height * width + c * height * width + h * width + w
                        // n * height * width * channel + h * width * channel + w * channel + c
                        if (NHWC2NCHW == dir) {
                            // nhwc -> nchw
                            dst[n * channel * height * width + c * height * width + h * width + w] =
                                src[n * height * width * channel + h * width * channel + w * channel + c];
                        } else {
                            // nchw -> nhwc
                            dst[n * height * width * channel + h * width * channel + w * channel + c] =
                                src[n * channel * height * width + c * height * width + h * width + w];
                        }
                    }
                }
            }
        }
        if (alloc_mem) {
            memcpy(src, dst, num * channel * height * width * sizeof(T));
            delete[] dst;
        }
        return TNN_OK;
    }
    static char* GetBlobPtr(BlobHandle handle) {
        return static_cast<char *>(handle.base) + handle.bytes_offset; 
    }
    template <class T>
    static Status ConvertFromNCHWToNHWC(Blob *src, Blob *dst) {
        ASSERT(src != nullptr);
        const auto src_dims = src->GetBlobDesc().dims;
        ASSERT(src_dims.size() > 1);
        const int num     = src_dims[0];
        const int channel = src_dims.size() > 1 ? src_dims[1] : 1;
        const int height  = src_dims.size() > 2 ? src_dims[2] : 1;
        const int width   = src_dims.size() > 3 ? src_dims[3] : 1;
        T *src_data_ptr   = (T *)GetBlobPtr(src->GetHandle());
        T *dst_data_ptr   = dst == nullptr ? nullptr : (T *)GetBlobPtr(dst->GetHandle());

        auto status = ConvertBetweenNHWCAndNCHW<T>(src_data_ptr, dst_data_ptr, num, channel, height, width, NCHW2NHWC);
        return status;
    }

    template <class T>
    static Status ConvertFromNHWCToNCHW(Blob *src, Blob *dst) {
        ASSERT(src != nullptr);
        const auto src_dims = src->GetBlobDesc().dims;
        ASSERT(src_dims.size() > 1);
        const int num     = src_dims[0];
        const int channel = src_dims.size() > 1 ? src_dims[1] : 1;
        const int height  = src_dims.size() > 2 ? src_dims[2] : 1;
        const int width   = src_dims.size() > 3 ? src_dims[3] : 1;
        T *src_data_ptr   = (T *)GetBlobPtr(src->GetHandle());
        T *dst_data_ptr   = dst == nullptr ? nullptr : (T *)GetBlobPtr(dst->GetHandle());

        auto status = ConvertBetweenNHWCAndNCHW<T>(src_data_ptr, dst_data_ptr, num, channel, height, width, NHWC2NCHW);
        return status;
    }
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_UTILS_DATA_FORMAT_CONVERTER_H_
