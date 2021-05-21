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

#ifndef TNN_ARM_COMMON_H_
#define TNN_ARM_COMMON_H_

#include <iostream>

#include "tnn/core/blob_int8.h"
#include "tnn/core/macro.h"
#include "tnn/device/arm/acc/Float4.h"
#include "tnn/device/arm/acc/compute/compute.h"
#include "tnn/device/arm/acc/compute/compute_int8.h"
#include "tnn/device/arm/arm_util.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/bfp16_utils.h"
#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

#define NEON_KERNEL_EXTRA_LOAD (64)

#ifndef NEON_INT8CONV_TILE_HW
#define NEON_INT8CONV_TILE_HW (4)
#endif

#ifdef TNN_USE_NEON

#ifdef __aarch64__
#define VQMOVN_HIGH_S32_T(lows16, highs32) vqmovn_high_s32((lows16), (highs32))
#define VMOVL_HIGH_S16_T(a) vmovl_high_s16(a)
#define VCVTAQ_S32_F32(a) vcvtaq_s32_f32(a)
#define VPADDQ_S32(a, b) vpaddq_s32(a, b)

#else
#define VQMOVN_HIGH_S32_T(lows16, highs32) vcombine_s16((lows16), vqmovn_s32(highs32))
#define VMOVL_HIGH_S16_T(a) vmovl_s16(vget_high_s16(a))
// trick convert for float, only accurate when abs(a) < 1.5 * 2^22, assume ok
// magic number 12582912.0f will do rounding to nearest, ties to even
// but naive and aarch64 will do rounding to nearest, ties away from zero, it's not aligned
// ties away from zero: val + (val >= 0.f ? 0.5f : -0.5f)
// const float32x4_t kNeonClampNumberf = vdupq_n_f32(12582912.0f);
// const int32x4_t kNeonClampNumberi   = vdupq_n_s32(0x4B400000);
// #define VCVTAQ_S32_F32(a) (vsubq_s32(vreinterpretq_s32_f32(vaddq_f32(a, kNeonClampNumberf)), kNeonClampNumberi))
const float32x4_t kNeonA = vdupq_n_f32(0.5f);
const float32x4_t kNeonB = vdupq_n_f32(-0.5f);
const float32x4_t kNeonZero = vdupq_n_f32(0.f);
#define VCVTAQ_S32_F32(a) (vcvtq_s32_f32(vaddq_f32(a, vbslq_f32(vcgeq_f32(a, kNeonZero), kNeonA, kNeonB))))

#define VPADDQ_S32(a, b)                                                                                               \
    vcombine_s32(vpadd_s32(vget_low_s32(a), vget_high_s32(a)), vpadd_s32(vget_low_s32(b), vget_high_s32(b)))

#endif

#endif

#endif
