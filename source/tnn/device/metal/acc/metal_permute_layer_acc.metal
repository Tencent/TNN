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

kernel void permute_to_nhwc(const device ftype4 *src                  [[buffer(0)]],
                                                device ftype4 *dst                            [[buffer(1)]],
                                                constant MetalParams &params     [[buffer(2)]],
                                                uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice * params.batch)))
        return;
    
    int index_out = (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
    
    int index_batch = gid.z / params.output_slice;
    int output_slice = gid.z % params.output_slice;
    int4 index_channel = output_slice*4 + int4(0, 1, 2, 3);
    int index_height = gid.y;
    int index_width = gid.x;
    //n h w c
    
    int input_batch = index_batch;
    int input_slice = index_width/4;
    int4 input_height = index_channel;
    bool4 valid_position = input_height < params.input_height;
    input_height = clamp(input_height, 0, params.input_height-1);
    int input_width = index_height;
    
    int4 input_i = index_width % 4;
    int4 index_in = input_batch * params.input_slice * params.input_size +
    input_slice * params.input_size +
    input_height * params.input_width
    + input_width;
    
    ftype4 val = ftype4(
        src[index_in[0]][input_i[0]],
        src[index_in[1]][input_i[1]],
        src[index_in[2]][input_i[2]],
        src[index_in[3]][input_i[3]]
    );
    val = select(ftype4(0), val, valid_position);
    dst[index_out] = val;
}

kernel void permute_to_nhcw(const device ftype4 *src                  [[buffer(0)]],
                                                device ftype4 *dst                            [[buffer(1)]],
                                                constant MetalParams &params     [[buffer(2)]],
                                                uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice * params.batch)))
        return;
    
    int index_out = (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
    
    int index_batch = gid.z / params.output_slice;
    int output_slice = gid.z % params.output_slice;
    int4 index_channel = output_slice*4 + int4(0, 1, 2, 3);
    int index_height = gid.y;
    int index_width = gid.x;
    //n h c w
    
    int input_batch = index_batch;
    int input_slice = index_height/4;
    int4 input_height = index_channel;
    bool4 valid_position = input_height < params.input_height;
    input_height = clamp(input_height, 0, params.input_height-1);
    int input_width = index_width;
    
    int4 input_i = index_height % 4;
    int4 index_in = input_batch * params.input_slice * params.input_size +
    input_slice * params.input_size +
    input_height * params.input_width
    + input_width;
    
    ftype4 val = ftype4(
        src[index_in[0]][input_i[0]],
        src[index_in[1]][input_i[1]],
        src[index_in[2]][input_i[2]],
        src[index_in[3]][input_i[3]]
    );
    val = select(ftype4(0), val, valid_position);
    dst[index_out] = val;
}

kernel void permute_to_nwch(const device ftype4 *src                  [[buffer(0)]],
                                                device ftype4 *dst                            [[buffer(1)]],
                                                constant MetalParams &params     [[buffer(2)]],
                                                uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice * params.batch)))
        return;

    int index_out = (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;

    int index_batch = gid.z / params.output_slice;
    int output_slice = gid.z % params.output_slice;
    int4 index_channel = output_slice*4 + int4(0, 1, 2, 3);
    int index_height = gid.y;
    int index_width = gid.x;
    //n w c h

    int input_batch = index_batch;
    int input_slice = index_height/4;
    int input_height = index_width;
    int4 input_width = index_channel;
    bool4 valid_position = input_width < params.input_width;
    input_width = clamp(input_width, 0, params.input_width-1);

    int4 input_i = index_height % 4;
    int4 index_in = input_batch * params.input_slice * params.input_size +
    input_slice * params.input_size +
    input_height * params.input_width
    + input_width;

    ftype4 val = ftype4(
        src[index_in[0]][input_i[0]],
        src[index_in[1]][input_i[1]],
        src[index_in[2]][input_i[2]],
        src[index_in[3]][input_i[3]]
    );
    val = select(ftype4(0), val, valid_position);
    dst[index_out] = val;
}

kernel void permute_to_chwn(const device ftype4 *src                  [[buffer(0)]],
                                                device ftype4 *dst                            [[buffer(1)]],
                                                constant MetalParams &params     [[buffer(2)]],
                                                uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice * params.batch)))
        return;

    int index_out = (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;

    int index_batch = gid.z / params.output_slice;
    int output_slice = gid.z % params.output_slice;
    int4 index_channel = output_slice*4 + int4(0, 1, 2, 3);
    int index_height = gid.y;
    int index_width = gid.x;
    //c h w n

    int input_batch = index_width;
    int input_slice = index_batch/4;
    int4 input_height = index_channel;
    bool4 valid_position = input_height < params.input_height;
    input_height = clamp(input_height, 0, params.input_height-1);
    int input_width = index_height;

    int4 input_i = index_batch % 4;
    int4 index_in = input_batch * params.input_slice * params.input_size +
    input_slice * params.input_size +
    input_height * params.input_width
    + input_width;

    ftype4 val = ftype4(
        src[index_in[0]][input_i[0]],
        src[index_in[1]][input_i[1]],
        src[index_in[2]][input_i[2]],
        src[index_in[3]][input_i[3]]
    );
    val = select(ftype4(0), val, valid_position);
    dst[index_out] = val;
}

kernel void permute_copy(const device ftype4 *src                  [[buffer(0)]],
                            device ftype4 *dst                            [[buffer(1)]],
                            constant MetalParams &params     [[buffer(2)]],
                            uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice * params.batch)))
        return;

    int index_in_out = (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
    dst[index_in_out] = src[index_in_out];
}
