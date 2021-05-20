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

kernel void splitv_axis_2(const device ftype4 *src                 [[buffer(0)]],
                                device ftype4 *dst                 [[buffer(1)]],
                          constant MetalParams &params             [[buffer(2)]],
                          constant int &h_offset                   [[buffer(3)]],
                          constant int &split_axis_size               [[buffer(4)]],
                          uint3 gid                                [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, split_axis_size, params.batch*params.output_slice)))
        return;

    auto output_size = params.output_width * split_axis_size;
    int index_out = (int)gid.z*output_size + (int)gid.y*params.output_width + (int)gid.x;

    int input_h = h_offset + int(gid.y);
    int input_w = int(gid.x);
    int index_in = (int)gid.z * params.input_size + input_h * params.input_width + input_w;

    dst[index_out] = src[index_in];
}

kernel void splitv_axis_3(const device ftype4 *src                 [[buffer(0)]],
                                device ftype4 *dst                 [[buffer(1)]],
                          constant MetalParams &params             [[buffer(2)]],
                          constant int &w_offset                   [[buffer(3)]],
                          constant int &split_axis_size               [[buffer(4)]],
                          uint3 gid                                [[thread_position_in_grid]]) {
    if (any(gid >= uint3(split_axis_size, params.output_height, params.batch*params.output_slice)))
        return;

    auto output_size = params.output_height * split_axis_size;
    int index_out = (int)gid.z*output_size + (int)gid.y*split_axis_size + (int)gid.x;

    int input_w = w_offset + int(gid.x);
    int input_h = int(gid.y);
    int index_in = (int)gid.z * params.input_size + input_h * params.input_width + input_w;

    dst[index_out] = src[index_in];
}


kernel void splitv_axis_1_common(const device ftype4 *src                       [[buffer(0)]],
                                    device ftype4 *dst                            [[buffer(1)]],
                                    constant MetalSplitVParamV2 &params     [[buffer(2)]],
                                    constant int &axis_offset           [[buffer(3)]],
                                    constant int &axis_size               [[buffer(4)]],
                                    uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.inner_size, axis_size, params.outer_size)))
        return;
    
    int index_out = (int)gid.z*axis_size*params.inner_size + (int)gid.y*params.inner_size + (int)gid.x;
    
    const int input_channel_count = params.axis_size*4;
    int4 input_channeles = (int)gid.y*4 + int4(0, 1, 2, 3) + axis_offset;
    int4 input_slice = input_channeles / 4;
    input_slice = min(input_slice, params.axis_size-1);
    int4 input_i = input_channeles % 4;
    
    int4 index_in = (int)gid.z*params.axis_size*params.inner_size + input_slice*params.inner_size + (int)gid.x;
    
    if (all( index_in == index_in.yzwx) &&
        all( input_i == int4(0, 1, 2, 3)) &&
        all( input_channeles < int4(input_channel_count)) ) {
        dst[index_out] = src[index_in[0]];
    } else {
        dst[index_out] = ftype4(
            src[index_in[0]][input_i[0]],
            src[index_in[1]][input_i[1]],
            src[index_in[2]][input_i[2]],
            src[index_in[3]][input_i[3]]
        );
    }
}

kernel void splitv_common(const device ftype4 *src                       [[buffer(0)]],
                            device ftype4 *dst                            [[buffer(1)]],
                            constant MetalSplitVParamV2 &params     [[buffer(2)]],
                            constant int &axis_offset           [[buffer(3)]],
                            constant int &axis_size               [[buffer(4)]],
                            uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.inner_size, axis_size, params.outer_size)))
        return;
    
    int index_out = (int)gid.z*axis_size*params.inner_size + (int)gid.y*params.inner_size + (int)gid.x;
    
    int input_axis_offset = axis_offset + (int)gid.y;
    
    int index_in = (int)gid.z*params.axis_size*params.inner_size + input_axis_offset*params.inner_size + (int)gid.x;
    
    dst[index_out] = src[index_in];
}