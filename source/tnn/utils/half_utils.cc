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

#include "tnn/utils/half_utils_inner.h"

#include "tnn/core/macro.h"

#if defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#import <Accelerate/Accelerate.h>
#endif
#endif

using namespace half_float;

namespace TNN_NS {

const float MAX_HALF_FLOAT = 65504.0f;
const float MIN_HALF_FLOAT = -65504.0f;

int ConvertFromFloatToHalf(float *fp32, void *fp16, int count) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    vImage_Buffer halfImage, floatImage;
    {
        halfImage.width    = count;
        halfImage.height   = 1;
        halfImage.rowBytes = count * sizeof(float) / 2;
        halfImage.data     = fp16;

        floatImage.width    = count;
        floatImage.height   = 1;
        floatImage.rowBytes = count * sizeof(float);
        floatImage.data     = fp32;
    }

    auto error = vImageConvert_PlanarFtoPlanar16F(&floatImage, &halfImage, 0);
    if (error != kvImageNoError) {
        LOGE("vImageConvert_PlanarFtoPlanar16F error\n");
        return -1;
    } else {
        return 0;
    }
#else
    bool exceedUplimits     = false;
    detail::uint16 *fp16PTR = (detail::uint16 *)fp16;
    for (int i = 0; i < count; ++i) {
        if (fp32[i] > MAX_HALF_FLOAT) {
            exceedUplimits = true;
            LOGE(
                "ERROR: the weights[%d]=%f of conv_layer_data is out of bounds "
                "of float16 max %f. \n",
                i, fp32[i], MAX_HALF_FLOAT);
            fp16PTR[i] = detail::float2half<(std::float_round_style)(HALF_ROUND_STYLE)>(MAX_HALF_FLOAT);
        } else if (fp32[i] < MIN_HALF_FLOAT) {
            exceedUplimits = true;
            LOGE(
                "ERROR: the weights[%d]=%f of conv_layer_data is out of bounds "
                "of float16 min %f. \n",
                i, fp32[i], MIN_HALF_FLOAT);
            fp16PTR[i] = detail::float2half<(std::float_round_style)(HALF_ROUND_STYLE)>(MIN_HALF_FLOAT);
            ;
        } else {
            fp16PTR[i] = detail::float2half<(std::float_round_style)(HALF_ROUND_STYLE)>(fp32[i]);
        }
    }
    return exceedUplimits ? -1 : 0;
#endif
}

int ConvertFromHalfToFloat(void *fp16, float *fp32, int count) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    vImage_Buffer halfImage, floatImage;
    {
        halfImage.width    = count;
        halfImage.height   = 1;
        halfImage.rowBytes = count * sizeof(float) / 2;
        halfImage.data     = fp16;

        floatImage.width    = count;
        floatImage.height   = 1;
        floatImage.rowBytes = count * sizeof(float);
        floatImage.data     = fp32;
    }

    auto error = vImageConvert_Planar16FtoPlanarF(&halfImage, &floatImage, 0);
    if (error != kvImageNoError) {
        LOGE("vImageConvert_Planar16FtoPlanarF error\n");
        return -1;
    } else {
        return 0;
    }
#else
    detail::uint16 *fp16PTR = (detail::uint16 *)fp16;
    for (int i = 0; i < count; ++i) {
        fp32[i] = detail::half2float<float>(fp16PTR[i]);
    }

    return 0;
#endif
}

}  // namespace TNN_NS
