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

#include "tnn/device/metal/acc/metal_common.h"
#include <metal_stdlib>

using namespace metal;

#if TNN_METAL_FULL_PRECISION
typedef float    ftype;
typedef float2   ftype2;
typedef float3   ftype3;
typedef float4   ftype4;
typedef float2x2 ftype2x2;
typedef float2x3 ftype2x3;
typedef float2x4 ftype2x4;
typedef float3x2 ftype3x2;
typedef float3x3 ftype3x3;
typedef float3x4 ftype3x4;
typedef float4x2 ftype4x2;
typedef float4x3 ftype4x3;
typedef float4x4 ftype4x4;
#define FTYPE_MAX MAXFLOAT
#else
typedef half     ftype;
typedef half2    ftype2;
typedef half3    ftype3;
typedef half4    ftype4;
typedef half2x2  ftype2x2;
typedef half2x3  ftype2x3;
typedef half2x4  ftype2x4;
typedef half3x2  ftype3x2;
typedef half3x3  ftype3x3;
typedef half3x4  ftype3x4;
typedef half4x2  ftype4x2;
typedef half4x3  ftype4x3;
typedef half4x4  ftype4x4;
#define FTYPE_MAX MAXHALF
#endif


using namespace metal;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-const-variable"
static constant ftype4 Zero4 = ftype4(0, 0, 0, 0);
static constant ftype4 One4 = ftype4(1, 1, 1, 1);
static constant ftype4 Six4 = ftype4(6, 6, 6, 6);
#pragma clang diagnostic pop


inline ftype4 prelu(ftype4 x, ftype4 slop) {
    return fmax(Zero4, x) + slop * fmin(Zero4, x);
}

inline ftype4 activate(ftype4 value, int type) {
    switch (type) {
        case 0x0001: // Relu see layer_param.h
            return max(value, Zero4);
        case 0x0002: // Relu6 see layer_param.h
            return clamp(value, Zero4, Six4);
        case 0x0100: // Sigmoid_Mul see layer_param.h
            return One4 / (One4 + exp(-value)) * value;
        default: // None
            return value;
    }
}

// compute tanh according to its definition
// metal::tanh may produce nan
inline ftype4 tanh_high_precision(ftype4 x) {
    float4 ep = exp(float4(x));
    float4 en = exp(float4(-x));
    float4 numerator   = ep - en;
    float4 denominator = ep + en;
    float4 result = numerator / denominator;

    return ftype4(result);
}

// compute softplus according to its definition
// metal::softplus may produce nan
inline ftype4 softplus_high_precision(ftype4 x) {
    float4 result = log(exp(float4(x)) + float4(One4));

    return ftype4(result);
}
