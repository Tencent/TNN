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

#ifndef TNN_ARM_UTIL_H_
#define TNN_ARM_UTIL_H_

#include <string.h>
#include <sys/time.h>

#include <cstdlib>

#include "tnn/core/blob.h"
#include "tnn/core/macro.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/half_utils_inner.h"

namespace TNN_NS {
namespace arm {
#if TNN_PROFILE
struct Timer {
public:
    void Start() {
        gettimeofday(&start, NULL);
    }
    float TimeEclapsed() {
        struct timeval end;
        gettimeofday(&end, NULL);
        float delta = (end.tv_sec - start.tv_sec) * 1000.f + (end.tv_usec - start.tv_usec) / 1000.f;
        gettimeofday(&start, NULL);
        return delta;
    }

private:
    struct timeval start;
};
#endif

char *GetBlobHandlePtr(BlobHandle handle);

template <typename Tin, typename Tout>
int PackC4(Tout *dst, const Tin *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int PackNHWC4(Tout *dst, const Tin *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int PackNHWC4FromNHWC(Tout *dst, const Tin *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int PackC8(Tout *dst, const Tin *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int PackCX(Tout *dst, const Tin *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int PackC4FromNHWC(Tout *dst, const Tin *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int PackC8FromNHWC(Tout *dst, const Tin *src, size_t hw, size_t channel);

int PackCAndQuant(int8_t *dst, const float *src, size_t hw, size_t channel, float *scale);

template <typename Tin, typename Tout>
int UnpackC4(Tout *dst, const Tin *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int UnpackNHWC4(Tout *dst, const Tin *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int UnpackNHWC4ToNHWC(Tout *dst, const Tin *src, size_t hw, size_t channel);

bool FloatBlobCanIgnorePack(size_t channel, size_t hw);
bool HalfBlobCanIgnorePack(size_t channel, size_t hw);
int PackFloatBlob(float *dst, float *src, size_t batch, size_t channel, size_t hw);
int UnpackFloatBlob(float *dst, float *src, size_t batch, size_t channel, size_t hw);
int PackInt32Blob(int32_t *dst, int32_t *src, size_t batch, size_t channel, size_t hw);
int UnpackInt32Blob(int32_t *dst, int32_t *src, size_t batch, size_t channel, size_t hw);
int PackFloatBlob(bfp16_t *dst, bfp16_t *src, size_t batch, size_t channel, size_t hw);
int UnpackFloatBlob(bfp16_t *dst, bfp16_t *src, size_t batch, size_t channel, size_t hw);
int PackHalfBlob(fp16_t *dst, fp16_t *src, size_t batch, size_t channel, size_t hw);
int UnpackHalfBlob(fp16_t *dst, fp16_t *src, size_t batch, size_t channel, size_t hw);

template <typename Tin, typename Tout>
int UnpackC8(Tout *dst, const Tin *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int UnpackCX(Tout *dst, const Tin *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int UnpackC4ToNHWC(Tout *dst, const Tin *src, size_t hw, size_t channel);

template <typename Tin, typename Tout>
int UnpackC8ToNHWC(Tout *dst, const Tin *src, size_t hw, size_t channel);

int UnpackC4WithStride(float *dst, const float *src, size_t ih, size_t iw, size_t c_step, size_t w_step, size_t depth);

int UnpackAndDequant(float *dst, const int8_t *src, size_t hw, size_t channel, float *scale, float *bias);

int UnpackHWC4ToCHW(int8_t *dst, const int8_t *src, size_t channel, size_t hw);

template <typename T>
int ConvertWeightsC4ToC8(T *weight, int ic, int oc);

template <typename T>
int ConvertWeightsFromGOIHWToGOIHW16(T *src, T *dst, int group, int input_channel, int output_channel, int height,
                                     int width);

template <typename T>
int ConvertWeightsFromGIOHWToGOHWI16(T *src, T *dst, int group, int input_channel, int output_channel, int height,
                                     int width);

template <typename T>
int ConvertWeightsFromGIOHWToGOHWI64(const T *src, T *dst, int group, int input_channel, int output_channel, int height,
                                     int width);

template <typename T>
int ConvertWeightsFromOI3HWToOHW12(T *src, T *dst, int input_channel, int output_channel, int height, int width);

template <typename T>
int ConvertWeightsFromOI3HWToOHW24(const T *src, T *dst, int input_channel, int output_channel, int height, int width);

template <typename T>
int ConvertWeightsFromGOIHWToGOIHW64(const T *src, T *dst, int group, int input_channel, int output_channel, int height,
                                     int width);

int PackINT8Weight(int8_t *src, int8_t *dst, int group, int input_channel, int output_channel, int height, int width);

void NV12ToBGR(const unsigned char *nv12, unsigned char *bgr, int height, int width);

void NV21ToBGR(const unsigned char *nv21, unsigned char *bgr, int height, int width);

void NV12ToBGRA(const unsigned char *nv12, unsigned char *bgra, int height, int width);

void NV21ToBGRA(const unsigned char *nv21, unsigned char *bgra, int height, int width);

void BGRToGray(const unsigned char *bgr, unsigned char *gray, int height, int width);

void BGRAToGray(const unsigned char *bgra, unsigned char *gray, int height, int width);

void RGBToGray(const unsigned char *rgb, unsigned char *gray, int height, int width);

void RGBAToGray(const unsigned char *rgba, unsigned char *gray, int height, int width);

}  // namespace arm
}  // namespace TNN_NS

#endif
