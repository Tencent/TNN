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
#include "metal_common.metal"

using namespace metal;

kernel void normaliz_2_axis_1_common_channel_0(const device ftype4 *src                  [[buffer(0)]],
                                                                                device ftype4 *dst                             [[buffer(1)]],
                                                                                constant MetalNormalizeParams &params      [[buffer(2)]],
                                                                                uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, 1, params.batch)))
        return;
    
    int index = (int)gid.z*params.output_slice*params.output_size + (int)gid.x;
    auto const src_data = src + index;
    auto const dst_data = dst + index;
    
    //xx
    auto src_data_ptr = src_data;
    auto sum4 = float4(*src_data_ptr) * float4(*src_data_ptr);
    for (int s = 1; s < params.output_slice; s++) {
        src_data_ptr += params.output_size;
        sum4 += float4(*src_data_ptr) * float4(*src_data_ptr);
    }
    //sum
    float4 sum_01 = sum4 + sum4.yzwx;
    float4 sum_23 = sum4.zwxy + sum4.wxyz;
    sum4 = sum_01 + sum_23;
    
    //max - sqrt
    sum4 = max(sqrt(sum4), float4(params.epsilon));
    sum4 = 1.0f/sum4;
    
    //division
    src_data_ptr = src_data;
    auto dst_data_ptr = dst_data;
    for (int s = 0; s < params.output_slice; s++) {
        *dst_data_ptr = ftype4(float4(*src_data_ptr)*sum4);
        dst_data_ptr += params.output_size;
        src_data_ptr += params.output_size;
    }
}

kernel void normaliz_2_axis_1_slice_1_channel_4(const device ftype4 *src                  [[buffer(0)]],
                                                                            device ftype4 *dst                             [[buffer(1)]],
                                                                            constant MetalNormalizeParams &params      [[buffer(2)]],
                                                                            uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    int index = (int)gid.z * params.output_size + (int)gid.x;
    auto input_data  = float4(src[index]);
    
    //x*x
    auto xx = input_data * input_data;
    
    //sum
    float4 sum_01 = xx + xx.yzwx;
    float4 sum_23 = xx.zwxy + xx.wxyz;
    float4 sum_0123 = sum_01 + sum_23;
    
    //max - sqrt
    sum_0123 = max(sqrt(sum_0123), params.epsilon);
    
    //div
    dst[index] = ftype4(input_data/sum_0123);
}

kernel void normaliz_2_axis_1_slice_1_channel_3(const device ftype4 *src                  [[buffer(0)]],
                                                                            device ftype4 *dst                             [[buffer(1)]],
                                                                            constant MetalNormalizeParams &params      [[buffer(2)]],
                                                                            uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    int index = (int)gid.z * params.output_size + (int)gid.x;
    auto input_data  = float4(src[index]);
    
    //x*x
    auto xx = input_data * input_data;
    
    //sum
    auto sum_012 = xx + xx.yzxw + xx.zxyw;
    
    //max - sqrt
    sum_012 = max(sqrt(sum_012), params.epsilon);
    
    //div
    dst[index] = ftype4(input_data/sum_012);
}

kernel void normaliz_2_axis_1_slice_1_channel_2(const device ftype4 *src                  [[buffer(0)]],
                                                                            device ftype4 *dst                             [[buffer(1)]],
                                                                            constant MetalNormalizeParams &params      [[buffer(2)]],
                                                                            uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    int index = (int)gid.z * params.output_size + (int)gid.x;
    auto input_data  = float4(src[index]);
    
    //x*x
    auto xx = input_data * input_data;
    
    //sum
    auto sum_01 = xx + xx.yxzw;
    
    //max - sqrt
    sum_01 = max(sqrt(sum_01), params.epsilon);
    
    //div
    dst[index] = ftype4(input_data/sum_01);
}


kernel void normaliz_1_axis_1_common_channel_0(const device ftype4 *src                  [[buffer(0)]],
                                                                                device ftype4 *dst                             [[buffer(1)]],
                                                                                constant MetalNormalizeParams &params      [[buffer(2)]],
                                                                                uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, 1, params.batch)))
        return;
    
    int index = (int)gid.z*params.output_slice*params.output_size + (int)gid.x;
    auto const src_data = src + index;
    auto const dst_data = dst + index;
    
    //sum abs
    auto src_data_ptr = src_data;
    auto sum4 = abs(float4(*src_data_ptr));
    for (int s = 1; s < params.output_slice; s++) {
        src_data_ptr += params.output_size;
        sum4 += abs(float4(*src_data_ptr));
    }
    auto sum_01 = sum4 + sum4.yzwx;
    auto sum_23 = sum4.zwxy + sum4.wxyz;
    sum4 = sum_01 + sum_23;
    sum4 = 1.0f/sum4;
    
    //division
    src_data_ptr = src_data;
    auto dst_data_ptr = dst_data;
    for (int s = 0; s < params.output_slice; s++) {
        *dst_data_ptr = ftype4(float4(*src_data_ptr)*sum4);
        dst_data_ptr += params.output_size;
        src_data_ptr += params.output_size;
    }
}

kernel void normaliz_1_axis_1_slice_1_channel_4(const device ftype4 *src                  [[buffer(0)]],
                                                                            device ftype4 *dst                             [[buffer(1)]],
                                                                            constant MetalNormalizeParams &params      [[buffer(2)]],
                                                                            uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    int index = (int)gid.z * params.output_size + (int)gid.x;
    auto input_data  = float4(src[index]);
    
    //abs
    auto xx = abs(input_data);
    
    //sum
    float4 sum_01 = xx + xx.yzwx;
    float4 sum_23 = xx.zwxy + xx.wxyz;
    float4 sum_0123 = sum_01 + sum_23;
    
    //div
    dst[index] = ftype4(input_data/sum_0123);
}

kernel void normaliz_1_axis_1_slice_1_channel_3(const device ftype4 *src                  [[buffer(0)]],
                                                                            device ftype4 *dst                             [[buffer(1)]],
                                                                            constant MetalNormalizeParams &params      [[buffer(2)]],
                                                                            uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    int index = (int)gid.z * params.output_size + (int)gid.x;
    auto input_data  = float4(src[index]);
    
    //abs
    auto xx = abs(input_data);
    
    //sum
    auto sum_012 = xx + xx.yzxw + xx.zxyw;
    
    //div
    dst[index] = ftype4(input_data/sum_012);
}

kernel void normaliz_1_axis_1_slice_1_channel_2(const device ftype4 *src                  [[buffer(0)]],
                                                                            device ftype4 *dst                             [[buffer(1)]],
                                                                            constant MetalNormalizeParams &params      [[buffer(2)]],
                                                                            uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    int index = (int)gid.z * params.output_size + (int)gid.x;
    auto input_data  = float4(src[index]);
    
    //abs
    auto xx = abs(input_data);
    
    //sum
    auto sum_01 = xx + xx.yxzw;
    
    //div
    dst[index] = ftype4(input_data/sum_01);
}



