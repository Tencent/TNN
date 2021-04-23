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

kernel void softmax_common(const device ftype4 *src     [[buffer(0)]],
                           device       ftype4 *dst     [[buffer(1)]],
                           constant MetalArgMaxOrMinParams &params  [[buffer(2)]],
                           uint3 gid                                [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.inner_size, 1, params.outer_size)))
        return;

    int index_in  = (int)gid.z * params.inner_size * params.reduce_size + (int)gid.x;

    ftype4 max4 = src[index_in];
    for(int i=1; i<params.reduce_size; ++i) {
        max4 = max(max4, src[index_in + i*params.inner_size]);
    }

    float4 sum4 = float4(Zero4);
    for(int i=0; i<params.reduce_size; ++i) {
        ftype4 val4 = src[index_in + i*params.inner_size];
        float4 rst4 = exp(float4(val4 - max4));
        dst[index_in + i*params.inner_size] = ftype4(rst4);
        sum4 += rst4;
    }

    sum4 = 1.0f / sum4;

    for(int i=0; i<params.reduce_size; ++i) {
        float4 val4 = float4(dst[index_in + i*params.inner_size]);
        val4 = val4 * sum4;
        dst[index_in + i*params.inner_size] = ftype4(val4);
    }
}

kernel void softmax_axis_2_common(
                                            const device ftype4 *src                [[buffer(0)]],
                                            device ftype4 *dst                      [[buffer(1)]],
                                            constant MetalSoftmaxParams &params            [[buffer(2)]],
                                            uint3 gid                               [[thread_position_in_grid]])
{
    if (any(gid >= uint3(params.output_width, 1, params.output_slice*params.batch)))
        return;
    
    int index = (int)gid.z*params.output_size + (int)gid.x;
    auto const src_data = src + index;
    auto const dst_data = dst + index;
    
    //max
    auto src_data_ptr = src_data;
    auto max4 = *src_data_ptr;
    for (int s = 1; s < params.output_height; s++) {
        src_data_ptr += params.output_width;
        max4 = max(max4, *src_data_ptr);
    }
    
    //exp
    src_data_ptr = src_data;
    auto dst_data_ptr = dst_data;
    float4 sum4 = float4(0,0,0,0);
    for (int s = 0; s < params.output_height; s++) {
        auto temp = exp(float4(*src_data_ptr - max4));
        *dst_data_ptr = ftype4(temp);
        sum4 += temp;
        
        src_data_ptr += params.output_width;
        dst_data_ptr += params.output_width;
    }
    
    //sum
    sum4 = 1.0f/sum4;
    
    //division
    dst_data_ptr = dst_data;
    for (int s = 0; s < params.output_height; s++) {
        *dst_data_ptr = ftype4(float4(*dst_data_ptr)*sum4);
        dst_data_ptr += params.output_width;
    }
}

kernel void softmax_axis_1_common_mode_0(
                                            const device ftype4 *src                [[buffer(0)]],
                                            device ftype4 *dst                      [[buffer(1)]],
                                            constant MetalSoftmaxParams &params            [[buffer(2)]],
                                            uint3 gid                               [[thread_position_in_grid]])
{
    if (any(gid >= uint3(params.output_size, 1, params.batch)))
        return;
    
    int index = (int)gid.z*params.output_slice*params.output_size + (int)gid.x;
    auto const src_data = src + index;
    auto const dst_data = dst + index;
    
    //max
    auto src_data_ptr = src_data;
    auto max4 = *src_data_ptr;
    for (int s = 1; s < params.output_slice; s++) {
        src_data_ptr += params.output_size;
        max4 = fmax(max4, *src_data_ptr);
    }
    auto max_01 = fmax(max4, max4.yzwx);
    auto max_23 = fmax(max4.zwxy, max4.wxyz);
    max4 = fmax(max_01, max_23);
    
    //exp
    src_data_ptr = src_data;
    auto dst_data_ptr = dst_data;
    float4 sum4 = float4(0,0,0,0);
    for (int s = 0; s < params.output_slice; s++) {
        auto temp = exp(float4(*src_data_ptr - max4));
        *dst_data_ptr = ftype4(temp);
        sum4 += temp;
        
        src_data_ptr += params.output_size;
        dst_data_ptr += params.output_size;
    }
    
    //sum
    float4 sum_01 = sum4 + sum4.yzwx;
    float4 sum_23 = sum4.zwxy + sum4.wxyz;
    sum4 = sum_01 + sum_23;
    sum4 = 1.0f/sum4;
    
    //division
    dst_data_ptr = dst_data;
    for (int s = 0; s < params.output_slice; s++) {
        *dst_data_ptr = ftype4(float4(*dst_data_ptr)*sum4);
        dst_data_ptr += params.output_size;
    }
}

kernel void softmax_axis_1_common(
                                            const device ftype4 *src                [[buffer(0)]],
                                            device ftype4 *dst                      [[buffer(1)]],
                                            constant MetalSoftmaxParams &params            [[buffer(2)]],
                                            uint3 gid                               [[thread_position_in_grid]])
{
    if (any(gid >= uint3(params.output_size, 1, params.batch)))
        return;
    
    int index = (int)gid.z*params.output_slice*params.output_size + (int)gid.x;
    auto const src_data = src + index;
    auto const dst_data = dst + index;
    
    int low_slice = params.channel_remain > 0 ? params.output_slice-1 : params.output_slice;
    //max
    auto src_data_ptr = src_data;
    auto max4 = *src_data_ptr;
    for (int s = 1; s < low_slice; s++) {
        src_data_ptr += params.output_size;
        max4 = max(max4, *src_data_ptr);
    }
    if (params.channel_remain == 1) {
        auto max4_remain = *(src_data_ptr + params.output_size);
        max4 = max(max4, max4_remain.xxxx);
    } else if (params.channel_remain == 2) {
        auto max4_remain = *(src_data_ptr + params.output_size);
        max4 = max(max4, max4_remain.xyxy);
    } else if (params.channel_remain == 3) {
        auto max4_remain = *(src_data_ptr + params.output_size);
        max4 = max(max4, max4_remain.xyzx);
    }
    auto max_01 = max(max4, max4.yzwx);
    auto max_23 = max(max4.zwxy, max4.wxyz);
    max4 = max(max_01, max_23);
    
    //exp
    src_data_ptr = src_data;
    auto dst_data_ptr = dst_data;
    float4 sum4 = float4(0,0,0,0);
    for (int s = 0; s < low_slice; s++) {
        auto temp = exp(float4(*src_data_ptr - max4));
        *dst_data_ptr = ftype4(temp);
        sum4 += temp;
        
        src_data_ptr += params.output_size;
        dst_data_ptr += params.output_size;
    }
    if (params.channel_remain == 1) {
        auto temp = exp(float4(*src_data_ptr - max4));
        temp = float4(temp.x, 0, 0, 0);
        *dst_data_ptr = ftype4(temp);
        sum4 += temp;
    } else if (params.channel_remain == 2) {
        auto temp = exp(float4(*src_data_ptr - max4));
        temp = float4(temp.xy, 0, 0);
        *dst_data_ptr = ftype4(temp);
        sum4 += temp;
    } else if (params.channel_remain == 3) {
        auto temp = exp(float4(*src_data_ptr - max4));
        temp = float4(temp.xyz, 0);
        *dst_data_ptr = ftype4(temp);
        sum4 += temp;
    }
    
    //sum
    float4 sum_01 = sum4 + sum4.yzwx;
    float4 sum_23 = sum4.zwxy + sum4.wxyz;
    sum4 = sum_01 + sum_23;
    sum4 = 1.0f/sum4;
    
    //division
    dst_data_ptr = dst_data;
    for (int s = 0; s < params.output_slice; s++) {
        *dst_data_ptr = ftype4(float4(*dst_data_ptr)*sum4);
        dst_data_ptr += params.output_size;
    }
}

kernel void softmax_axis_1_slice_1_channel_4(
                                            const device ftype4 *src                [[buffer(0)]],
                                            device ftype4 *dst                      [[buffer(1)]],
                                            constant MetalSoftmaxParams &params            [[buffer(2)]],
                                            uint3 gid                               [[thread_position_in_grid]])
{
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    int index = (int)gid.z * params.output_size + (int)gid.x;
    auto in  = src[index];
    
    //max
    auto max_01 = fmax(in, in.yzwx);
    auto max_23 = fmax(in.zwxy, in.wxyz);
    auto max_0123 = fmax(max_01, max_23);
    
    //exp
    float4 exp4 = exp(float4(in - max_0123));
    
    //sum
    float4 sum_01 = exp4 + exp4.yzwx;
    float4 sum_23 = exp4.zwxy + exp4.wxyz;
    float4 sum_0123 = sum_01 + sum_23;
    
    //division
    dst[index] = ftype4(exp4/sum_0123);
}

kernel void softmax_axis_1_slice_1_channel_3(
                                             const device ftype4 *src                [[buffer(0)]],
                                             device ftype4 *dst                      [[buffer(1)]],
                                             constant MetalSoftmaxParams &params            [[buffer(2)]],
                                             uint3 gid                               [[thread_position_in_grid]])
 {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    int index = (int)gid.z * params.output_size + (int)gid.x;
    auto in  = src[index];
    
    //max
    auto max_012 = fmax(fmax(in, in.yzxw), in.zxyw);
    
    //exp
    float4 exp4 = exp(float4(in - max_012));
    
    //sum
    float4 sum_012 = exp4 + exp4.yzxw + exp4.zxyw;
    
    //division
    float4 div = exp4/sum_012;
    dst[index] = ftype4(ftype3(div.xyz), 0.0h);
}

kernel void softmax_axis_1_slice_1_channel_2(
                                             const device ftype4 *src                [[buffer(0)]],
                                             device ftype4 *dst                      [[buffer(1)]],
                                             constant MetalSoftmaxParams &params            [[buffer(2)]],
                                             uint3 gid                               [[thread_position_in_grid]])
{
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    int index = (int)gid.z * params.output_size + (int)gid.x;
    auto in  = src[index];
    
    //max
    auto max_01 = fmax(in, in.yxzw);
    
    //exp
    float4 exp4 = exp(float4(in - max_01));
    
    //sum
    float4 sum_012 = exp4 + exp4.yxzw;
    
    //division
    float4 div = exp4/sum_012;
    dst[index] = ftype4(ftype2(div.xy), 0.0h, 0.0h);
}
