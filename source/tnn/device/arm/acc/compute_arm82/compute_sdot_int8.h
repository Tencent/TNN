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

#include "tnn/core/macro.h"
#include "tnn/device/arm/arm_common.h"

namespace TNN_NS {

void PackSDOTINT8Weight(const int8_t *src, int8_t *dst, int oc, int ic, int kh, int kw);
void PackSDOTINT8WeightGemv(const int8_t *src, int8_t *dst, const int oc, const int ic, const int hw);
void PackSDOTDW3X3INT8Weight(const int8_t *src, int8_t *dst, int oc);

#ifdef TNN_ARM82_A32
void GemmInt8SdotUnit4x8(int8_t* dst, const int8_t* src, const int8_t* weight,
                         long src_depth, long dst_depth, long hw,
                         const int32_t* bias, const float* scale,
                         long relu, const int8_t* add_input,
                         const float* add_scale, const int8_t* relu6_max);
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef TNN_ARM82_USE_NEON
void ConvDw3x3Int8SdotSlideW(int8_t *dst_z, int8_t **src, const int8_t* weight_z, const int32_t* bias_z,
                             const float* scale_z, long dc, long dst_depth, long width);
void ConvDw3x3Int8SdotSlideWLeftC4(int8_t *dst_z, int8_t **src, const int8_t* weight_z, const int32_t* bias_z,
                             const float* scale_z, long dc, long dst_depth, long width);

void ConvDw3x3S2Int8SdotSlideW(int8_t *dst_z, int8_t **src, const int8_t* weight_z, const int32_t* bias_z,
                               const float* scale_z, long dc, long dst_depth, long width);
void ConvDw3x3S2Int8SdotSlideWLeftC4(int8_t *dst_z, int8_t **src, const int8_t* weight_z, const int32_t* bias_z,
                               const float* scale_z, long dc, long dst_depth, long width);

void GemvInt8Sdot(int8_t* dst, const int8_t* src, const int8_t* weight,
                  const int32_t* bias, const float* scale, long ic_r4, long oc_r4);
#endif

#if defined(TNN_ARM82_A32)
void GemmInt8SdotUnit4x8Kernel(int8_t* dst, const int8_t* src, const int8_t* weight,
                               long src_depth, long dst_depth, long hw,
                               const int32_t* bias, const float* scale,
                               long relu, const int8_t* add_input,
                               const float* add_scale, const int8_t* relu6_max);

void GemmInt8SdotUnit4x4(int8_t* dst, const int8_t* src, const int8_t* weight,
                         long src_depth, long dst_depth, long hw,
                         const int32_t* bias, const float* scale,
                         long relu, const int8_t* add_input,
                         const float* add_scale, const int8_t* relu6_max);
#elif defined(TNN_ARM82_A64)
void GemmInt8SdotUnit8x8(int8_t* dst, const int8_t* src, const int8_t* weight,
                        long src_depth, long dst_depth, long hw, 
                        const int32_t* bias, const float* scale,
                        long relu, const int8_t* add_input, 
                        const float* add_scale, const int8_t* relu6_max);

void GemmInt8SdotUnit8x4(int8_t* dst, const int8_t* src, const int8_t* weight,
                        long src_depth, long dst_depth, long hw, 
                        const int32_t* bias, const float* scale,
                        long relu, const int8_t* add_input, 
                        const float* add_scale, const int8_t* relu6_max);
#endif

#ifdef __cplusplus
}
#endif

}