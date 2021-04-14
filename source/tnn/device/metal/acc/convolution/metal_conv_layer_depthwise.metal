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

#include <metal_stdlib>
#include "tnn/device/metal/acc/metal_common.metal"

using namespace metal;

kernel void convolution_depthwise(const device ftype4 *in           [[buffer(0)]],
                                  device ftype4 *out                [[buffer(1)]],
                                  constant MetalConvParams& params  [[buffer(2)]],
                                  const device ftype4 *wt           [[buffer(3)]],
                                  const device ftype4 *biasTerms    [[buffer(4)]],
                                  uint3 gid                       [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width,
                           params.output_height,
                           params.batch*params.output_slice)))
        return;
    
    int oz = gid.z % params.output_slice;
    int offset_x = (int)gid.x * params.stride_x - params.pad_x;
    int offset_y = (int)gid.y * params.stride_y - params.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, params.dilation_x)));
    int ex = min(params.kernel_x, UP_DIV(params.input_width - offset_x, params.dilation_x));
    int sy = max(0, (UP_DIV(-offset_y, params.dilation_y)));
    int ey = min(params.kernel_y, UP_DIV(params.input_height - offset_y, params.dilation_y));
    offset_x += sx * params.dilation_x;
    offset_y += sy * params.dilation_y;
    
    auto z_wt  = wt  + (int)oz * params.kernel_size;
    auto z_in  = in  + (int)gid.z * params.input_size;
    auto z_out = out + (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
    
    auto result = params.has_bias ? float4(biasTerms[oz]) : float4(Zero4);
    for (auto ky = sy, y = offset_y; ky < ey; ky++, y += params.dilation_y) {
        for (auto kx = sx, x = offset_x; kx < ex; kx++, x += params.dilation_x) {
            auto wt4 = float4(z_wt[ky * params.kernel_x   + kx]);
            auto in4 = float4(z_in[ y * params.input_width + x]);
            result += in4 * wt4;
        }
    }
    
    *z_out = activate(ftype4(result), params.activation);
}

kernel void convolution_depthwise5x5_h8w4(const device ftype4 *in           [[buffer(0)]],
                                          device ftype4 *out                [[buffer(1)]],
                                          constant MetalConvParams& params  [[buffer(2)]],
                                          const device ftype4 *wt           [[buffer(3)]],
                                          const device ftype4 *biasTerms    [[buffer(4)]],
                                          uint3 gid                       [[thread_position_in_grid]],
                                          uint3 group_id                  [[threadgroup_position_in_grid]],
                                          uint thread_index               [[thread_index_in_threadgroup]]) {
    threadgroup ftype4 input_data_cache[8 * 12];
    
    // compute ld offset of inputs
    const int ld_start_w = group_id.x * 4 - params.pad_x;
    const int ld_start_h = group_id.y * 8 - params.pad_y;
    const int ld_start_c = group_id.z;
    
    const int ld_offset = ld_start_c * params.input_size;
    
    const int a_smem_st_offset = thread_index;
    
    // load data
    int ld_w = ld_start_w + thread_index % 8;
    int ld_h = ld_start_h + thread_index / 8;
    const int ld_pos = ld_offset + ld_h * params.input_width + ld_w;
    
    bool w_in_image = ld_w >=0 && ld_w < params.input_width;
    
    bool in_image = (ld_h >=0 && ld_h < params.input_height) && w_in_image;
    ftype4 v = in_image ? in[ld_pos] : Zero4;
    input_data_cache[a_smem_st_offset] = v;
    
    bool in_image1 =  ld_h + 4 >= 0 && ld_h + 4 < params.input_height && w_in_image ;
    v = in_image1 ? in[ld_pos + 4 * params.input_width] : Zero4;
    input_data_cache[a_smem_st_offset +  32] = v;
    
    bool in_image2 =  ld_h + 8 >= 0 && ld_h + 8 < params.input_height && w_in_image;
    v = in_image2 ? in[ld_pos + 8 * params.input_width] : Zero4;
    input_data_cache[a_smem_st_offset + 64] = v;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (!any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch))) {
        auto z_out = out + (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
        auto result = params.has_bias ? biasTerms[gid.z] : Zero4;
        auto z_wt  = wt  + (int)gid.z * params.kernel_size;
        
        int offset_x = thread_index % 4;
        int offset_y = thread_index / 4;
#pragma unroll
        for (auto ky = 0, y = offset_y; ky < 5; ky++, y ++) {
            for (auto kx = 0, x = offset_x; kx < 5; kx++, x ++) {
                auto wt4 = z_wt[ky * 5   + kx];
                auto in4 = input_data_cache[ y * 8 + x];
                result += in4 * wt4;
            }
        }
        
        *z_out = activate(result, params.activation);
    }
}

kernel void convolution_depthwise5x1_h8w4(const device ftype4 *in           [[buffer(0)]],
                                          device ftype4 *out                [[buffer(1)]],
                                          constant MetalConvParams& params  [[buffer(2)]],
                                          const device ftype4 *wt           [[buffer(3)]],
                                          const device ftype4 *biasTerms    [[buffer(4)]],
                                          uint3 gid                       [[thread_position_in_grid]],
                                          uint3 group_id                  [[threadgroup_position_in_grid]],
                                          uint thread_index               [[thread_index_in_threadgroup]]) {
    threadgroup ftype4 input_data_cache[8 * 8];
    
    // compute ld offset of inputs
    const int ld_start_w = group_id.x * 4 - params.pad_x;
    const int ld_start_h = group_id.y * 8 - params.pad_y;
    const int ld_start_c = group_id.z;
    
    const int ld_offset = ld_start_c * params.input_size;
    
    const int a_smem_st_offset = thread_index;
    
    // load data
    int ld_w = ld_start_w + thread_index % 8;
    int ld_h = ld_start_h + thread_index / 8;
    const int ld_pos = ld_offset + ld_h * params.input_width + ld_w;
    
    bool w_in_image = ld_w >=0 && ld_w < params.input_width;
    
    bool in_image = (ld_h >=0 && ld_h < params.input_height) && w_in_image;
    ftype4 v = in_image ? in[ld_pos] : Zero4;
    input_data_cache[a_smem_st_offset] = v;
    
    bool in_image1 =  ld_h + 4 >= 0 && ld_h + 4 < params.input_height && w_in_image ;
    v = in_image1 ? in[ld_pos + 4 * params.input_width] : Zero4;
    input_data_cache[a_smem_st_offset +  32] = v;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (!any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch))) {
        auto z_out = out + (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
        auto result = params.has_bias ? biasTerms[gid.z] : Zero4;
        auto z_wt  = wt  + (int)gid.z * params.kernel_size;
        
        int offset_x = thread_index % 4;
        int y = thread_index / 4;
#pragma unroll
        for (auto kx = 0, x = offset_x; kx < 5; kx++, x ++) {
            auto wt4 = z_wt[kx];
            auto in4 = input_data_cache[ y * 8 + x];
            result += in4 * wt4;
        }
        
        *z_out = activate(result, params.activation);
    }
}

kernel void convolution_depthwise1x5_h4w8(const device ftype4 *in           [[buffer(0)]],
                                          device ftype4 *out                [[buffer(1)]],
                                          constant MetalConvParams& params  [[buffer(2)]],
                                          const device ftype4 *wt           [[buffer(3)]],
                                          const device ftype4 *biasTerms    [[buffer(4)]],
                                          uint3 gid                       [[thread_position_in_grid]],
                                          uint3 group_id                  [[threadgroup_position_in_grid]],
                                          uint thread_index               [[thread_index_in_threadgroup]]) {
    threadgroup ftype4 input_data_cache[8 * 8];
    
    // compute ld offset of inputs
    const int ld_start_w = group_id.x * 8 - params.pad_x;
    const int ld_start_h = group_id.y * 4 - params.pad_y;
    const int ld_start_c = group_id.z;
    
    const int ld_offset = ld_start_c * params.input_size;
    
    const int a_smem_st_offset = thread_index;
    
    // load data
    int ld_w = ld_start_w + thread_index % 8;
    int ld_h = ld_start_h + thread_index / 8;
    const int ld_pos = ld_offset + ld_h * params.input_width + ld_w;
    
    bool w_in_image = ld_w >=0 && ld_w < params.input_width;
    
    bool in_image = (ld_h >=0 && ld_h < params.input_height) && w_in_image;
    ftype4 v = in_image ? in[ld_pos] : Zero4;
    input_data_cache[a_smem_st_offset] = v;
    
    bool in_image1 =  ld_h + 4 >= 0 && ld_h + 4 < params.input_height && w_in_image ;
    v = in_image1 ? in[ld_pos + 4 * params.input_width] : Zero4;
    input_data_cache[a_smem_st_offset +  32] = v;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (!any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch))) {
        auto z_out = out + (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
        auto result = params.has_bias ? biasTerms[gid.z] : Zero4;
        auto z_wt  = wt  + (int)gid.z * params.kernel_size;
        
        int x = thread_index % 8;
        int offset_y = thread_index / 8;
#pragma unroll
        for (auto ky = 0, y = offset_y; ky < 5; ky++, y ++) {
            auto wt4 = z_wt[ky];
            auto in4 = input_data_cache[ y * 8 + x];
            result += in4 * wt4;
        }
        
        *z_out = activate(result, params.activation);
    }
}
