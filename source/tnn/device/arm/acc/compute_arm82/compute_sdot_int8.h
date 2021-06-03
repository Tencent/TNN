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

void GEMM_SDOT_INT8_8X8(int8_t* dst, const int8_t* src, const int8_t* weight, long src_depth,
                       long dst_depth, long hw, const int32_t* bias, const float* scale,
                       long relu, const int8_t* add_input, const float* add_scale,
                       const int8_t* relu6_max);

void GEMM_SDOT_INT8_8X4(int8_t* dst, const int8_t* src, const int8_t* weight, long src_depth,
                       long dst_depth, long hw, const int32_t* bias, const float* scale,
                       long relu, const int8_t* add_input, const float* add_scale,
                       const int8_t* relu6_max);

void PackSDOTINT8Weight(const int8_t *src, int8_t *dst, int oc, int ic, int kh, int kw);

#ifdef TNN_ARM82_A64
#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif
#endif

}