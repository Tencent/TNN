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

#include <metal_math>
#include <metal_stdlib>
#include "tnn/device/metal/acc/metal_common.metal"

using namespace metal;

#define INPUT_PARAM const device ftype4 *src                [[buffer(0)]],              \
                    device ftype4 *dst                      [[buffer(1)]],              \
                    constant MetalReduceParams &params      [[buffer(2)]],              \
                    uint3 gid                               [[thread_position_in_grid]] \

#define DEFINE_REDUCE_AXIS_0(name, ini, op1, op2, post)  \
kernel void reduce_##name##_axis_0_common(INPUT_PARAM) { \
    if (any(gid >= uint3(params.output_size, params.output_slice, params.output_batch)))    \
        return;                                                                             \
    int step = params.input_slice * params.input_size;                                      \
    auto z_in  = src  + (int)gid.y * params.input_size + (int)gid.x;                        \
    auto z_out = dst + (int)gid.z * params.output_slice * params.output_size + (int)gid.y * params.output_size + (int)gid.x;    \
    auto s = ftype4(ini);                                                                                                       \
    for (int b = 0; b < params.input_batch; b++) {                                                                              \
        auto t = *(z_in + b * step);                                                                                            \
        t = op1;                                                                                                                \
        s = op2;                                                                                                                \
    }                                                                                                                           \
    s = post;                                                                                                                   \
    *z_out = s;                                                                                                                 \
} \

#define DEFINE_REDUCE_AXIS_1(name, ini, op1, op2, post)                                         \
kernel void reduce_##name##_axis_1_common(INPUT_PARAM) {                                        \
    if (any(gid >= uint3(params.output_size, params.output_slice, params.output_batch)))        \
        return;                                                                                 \
    int step = params.input_size;                                                               \
    auto z_in  = src  + (int)gid.z * params.input_slice * params.input_size + (int)gid.x;       \
    auto z_out = dst + (int)gid.z * params.output_slice * params.output_size + (int)gid.y * params.output_size + (int)gid.x;    \
    auto s = ftype4(ini);                                                                                                       \
    for (int b = 0; b < params.input_slice - 1; b++) {                                                                          \
        auto t = *(z_in + b * step);                                                                                            \
        t = op1;                                                                                                                \
        s = op2;                                                                                                                \
    }                                                                                                                           \
    auto t = *(z_in + step * (params.input_slice - 1));                                                                         \
    t = op1;                                                                                                                    \
    switch (params.input_channel_mode_4) {                                                                                      \
        case 1:                                                                                                                 \
            t.yzw = ini;                                                                                                        \
            break;                                                                                                              \
        case 2:                                                                                                                 \
            t.zw = ini;                                                                                                         \
            break;                                                                                                              \
        case 3:                                                                                                                 \
            t.w = ini;                                                                                                          \
            break;                                                                                                              \
    }                                                                                                                           \
    s = op2;                                                                                                                    \
    s = post;                                                                                                                   \
    *z_out = s;                                                                                                                 \
}                                                                                                                               \

#define DEFINE_REDUCE_AXIS_2(name, ini, op1, op2, post)                                         \
kernel void reduce_##name##_axis_2_common(INPUT_PARAM) {                                        \
    if (any(gid >= uint3(params.output_size, params.output_slice, params.output_batch)))        \
        return;                                                                                 \
    int step = params.input_width;                                                              \
    auto z_in  = src + (int)gid.z * params.input_slice * params.input_size + (int)gid.y * params.input_size + (int)gid.x;       \
    auto z_out = dst + (int)gid.z * params.output_slice * params.output_size + (int)gid.y * params.output_size + (int)gid.x;    \
    auto s = ftype4(ini);                                                                                                       \
    for (int b = 0; b < params.input_height; b++) {                                                                             \
        auto t = *(z_in + b * step);                                                                                            \
        t = op1;                                                                                                                \
        s = op2;                                                                                                                \
    }                                                                                                                           \
    s = post;                                                                                                                   \
    *z_out = s;                                                                                                                 \
}                                                                                                                               \

#define DEFINE_REDUCE_AXIS_3(name, ini, op1, op2, post)                                         \
kernel void reduce_##name##_axis_3_common(INPUT_PARAM) {                                        \
    if (any(gid >= uint3(params.output_size, params.output_slice, params.output_batch)))        \
        return;                                                                                 \
    auto z_in  = src + (int)gid.z * params.input_slice * params.input_size + (int)gid.y * params.input_size + (int)gid.x * params.input_width;  \
    auto z_out = dst + (int)gid.z * params.output_slice * params.output_size + (int)gid.y * params.output_size + (int)gid.x;                    \
    auto s = ftype4(ini);                                                                                                                       \
    for (int b = 0; b < params.input_width; b++) {                                                                                              \
        auto t = *(z_in + b);                                                                                                                   \
        t = op1;                                                                                                                                \
        s = op2;                                                                                                                                \
    }                                                                                                                                           \
    s = post;                                                                                                                                   \
    *z_out = s;                                                                                                                                 \
}

DEFINE_REDUCE_AXIS_0(l1, 0, abs(t), s+t, s);
DEFINE_REDUCE_AXIS_1(l1, 0, abs(t), s+t, dot(s,1));
DEFINE_REDUCE_AXIS_2(l1, 0, abs(t), s+t, s);
DEFINE_REDUCE_AXIS_3(l1, 0, abs(t), s+t, s);

DEFINE_REDUCE_AXIS_0(l2, 0, pow(t,2), s+t, sqrt(s));
DEFINE_REDUCE_AXIS_1(l2, 0, pow(t,2), s+t, sqrt(dot(s,1)));
DEFINE_REDUCE_AXIS_2(l2, 0, pow(t,2), s+t, sqrt(s));
DEFINE_REDUCE_AXIS_3(l2, 0, pow(t,2), s+t, sqrt(s));

DEFINE_REDUCE_AXIS_0(mean, 0, t, s+t, s/params.input_batch);
DEFINE_REDUCE_AXIS_1(mean, 0, t, s+t, dot(s,1)/params.input_channel);
DEFINE_REDUCE_AXIS_2(mean, 0, t, s+t, s/params.input_height);
DEFINE_REDUCE_AXIS_3(mean, 0, t, s+t, s/params.input_width);

DEFINE_REDUCE_AXIS_0(max, -FLT_MAX, t, max(t,s), s);
DEFINE_REDUCE_AXIS_1(max, -FLT_MAX, t, max(t,s), ftype4(max(max(s.x,s.y),max(s.z,s.w)),0,0,0));
DEFINE_REDUCE_AXIS_2(max, -FLT_MAX, t, max(t,s), s);
DEFINE_REDUCE_AXIS_3(max, -FLT_MAX, t, max(t,s), s);

DEFINE_REDUCE_AXIS_0(min, FLT_MAX, t, min(t,s), s);
DEFINE_REDUCE_AXIS_1(min, FLT_MAX, t, min(t,s), ftype4(min(min(s.x,s.y),min(s.z,s.w)),0,0,0));
DEFINE_REDUCE_AXIS_2(min, FLT_MAX, t, min(t,s), s);
DEFINE_REDUCE_AXIS_3(min, FLT_MAX, t, min(t,s), s);

DEFINE_REDUCE_AXIS_0(log_sum, 0, t, s+t, log(s));
DEFINE_REDUCE_AXIS_1(log_sum, 0, t, s+t, log(dot(s,1)));
DEFINE_REDUCE_AXIS_2(log_sum, 0, t, s+t, log(s));
DEFINE_REDUCE_AXIS_3(log_sum, 0, t, s+t, log(s));

DEFINE_REDUCE_AXIS_0(log_sum_exp, 0, exp(t), s+t, log(s));
DEFINE_REDUCE_AXIS_1(log_sum_exp, 0, exp(t), s+t, log(dot(s,1)));
DEFINE_REDUCE_AXIS_2(log_sum_exp, 0, exp(t), s+t, log(s));
DEFINE_REDUCE_AXIS_3(log_sum_exp, 0, exp(t), s+t, log(s));

DEFINE_REDUCE_AXIS_0(prod, 1, t, s*t, s);
DEFINE_REDUCE_AXIS_1(prod, 1, t, s*t, s.x*s.y*s.w*s.z);
DEFINE_REDUCE_AXIS_2(prod, 1, t, s*t, s);
DEFINE_REDUCE_AXIS_3(prod, 1, t, s*t, s);

DEFINE_REDUCE_AXIS_0(sum, 0, t, s+t, s);
DEFINE_REDUCE_AXIS_1(sum, 0, t, s+t, dot(s,1));
DEFINE_REDUCE_AXIS_2(sum, 0, t, s+t, s);
DEFINE_REDUCE_AXIS_3(sum, 0, t, s+t, s);

DEFINE_REDUCE_AXIS_0(sum_square, 0, pow(t,2), s+t, s);
DEFINE_REDUCE_AXIS_1(sum_square, 0, pow(t,2), s+t, dot(s,1));
DEFINE_REDUCE_AXIS_2(sum_square, 0, pow(t,2), s+t, s);
DEFINE_REDUCE_AXIS_3(sum_square, 0, pow(t,2), s+t, s);