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
                                                constant MetalPermuteParams &params     [[buffer(2)]],
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
    int4 index_in = input_batch * params.input_slice * params.input_size + \
                    input_slice * params.input_size + \
                    input_height * params.input_width + input_width;
    
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
                                                constant MetalPermuteParams &params     [[buffer(2)]],
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
    int4 index_in = input_batch * params.input_slice * params.input_size + \
                    input_slice * params.input_size + \
                    input_height * params.input_width + input_width;
    
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
                                                constant MetalPermuteParams &params     [[buffer(2)]],
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
    int4 index_in = input_batch * params.input_slice * params.input_size + \
                    input_slice * params.input_size + \
                    input_height * params.input_width + input_width;

    ftype4 val = ftype4(
        src[index_in[0]][input_i[0]],
        src[index_in[1]][input_i[1]],
        src[index_in[2]][input_i[2]],
        src[index_in[3]][input_i[3]]
    );
    val = select(ftype4(0), val, valid_position);
    dst[index_out] = val;
}

kernel void permute_to_nwhc(const device ftype4 *src                  [[buffer(0)]],
                                                device ftype4 *dst                            [[buffer(1)]],
                                                constant MetalPermuteParams &params     [[buffer(2)]],
                                                uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice * params.batch)))
        return;
    //0, 3, 2, 1
    int index_out = (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;

    int index_batch = gid.z / params.output_slice;
    int output_slice = gid.z % params.output_slice;
    int4 index_channel = output_slice*4 + int4(0, 1, 2, 3);
    int index_height = gid.y;
    int index_width = gid.x;
    //n w c h

    int input_batch = index_batch;
    int input_slice = index_width/4;
    int input_height = index_height;
    int4 input_width = index_channel;
    bool4 valid_position = input_width < params.input_width;
    input_width = clamp(input_width, 0, params.input_width-1);

    int4 input_i = index_width % 4;
    int4 index_in = input_batch * params.input_slice * params.input_size + \
                    input_slice * params.input_size + \
                    input_height * params.input_width + input_width;

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
                                                constant MetalPermuteParams &params     [[buffer(2)]],
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
    int4 index_in = input_batch * params.input_slice * params.input_size + \
                    input_slice * params.input_size + \
                    input_height * params.input_width + input_width;

    ftype4 val = ftype4(
        src[index_in[0]][input_i[0]],
        src[index_in[1]][input_i[1]],
        src[index_in[2]][input_i[2]],
        src[index_in[3]][input_i[3]]
    );
    val = select(ftype4(0), val, valid_position);
    dst[index_out] = val;
}

kernel void permute_to_wnch(const device ftype4 *src                  [[buffer(0)]],
                                                device ftype4 *dst                            [[buffer(1)]],
                                                constant MetalPermuteParams &params     [[buffer(2)]],
                                                uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice * params.batch)))
        return;

    int index_out = (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;

    int index_batch = gid.z / params.output_slice;
    int output_slice = gid.z % params.output_slice;
    int4 index_channel = output_slice*4 + int4(0, 1, 2, 3);
    int index_height = gid.y;
    int index_width = gid.x;
    //w n c h

    int4 input_batch = index_channel;
    int input_slice = index_height / 4;
    int input_height = index_width;
    int input_width = index_batch;
    bool4 valid_position = input_batch < params.batch;
    input_batch = clamp(input_batch, 0, params.input_batch-1);

    int4 input_i = index_height % 4;
    int4 index_in = input_batch * params.input_slice * params.input_size + \
                    input_slice * params.input_size + \
                    input_height * params.input_width + input_width;

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
                            constant MetalPermuteParams &params     [[buffer(2)]],
                            uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice * params.batch)))
        return;

    int index_in_out = (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
    dst[index_in_out] = src[index_in_out];
}

kernel void permute_copy_int4(const device int4 *src                  [[buffer(0)]],
                            device int4 *dst                            [[buffer(1)]],
                            constant MetalPermuteParams &params     [[buffer(2)]],
                            uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice * params.batch)))
        return;

    int index_in_out = (int)gid.z * params.output_size + (int)gid.y * params.output_width + (int)gid.x;
    dst[index_in_out] = src[index_in_out];
}

kernel void permute_common(const device ftype4 *src                  [[buffer(0)]],
                           device ftype4 *dst                        [[buffer(1)]],
                           constant MetalDynamicPermuteParams &params       [[buffer(2)]],
                           uint3 gid                                 [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;

    int index_out = ((int)gid.z * params.output_slice + (int)gid.y) * params.output_size + (int)gid.x;

    int batch  = (int)gid.z;
    int slice  = (int)gid.y;
    int4 channel = slice * 4 + int4(0, 1, 2, 3);
    int size  = (int)gid.x;
    bool4 valid_position = channel < params.channel_dim_size;
    channel = clamp(channel, 0, params.channel_dim_size-1);

    int4 input_i  = int4(0);
    int input_slice = 0;
    int4 index_in = 0;

    if (params.channel_dim == 0) {
        input_slice = batch / 4;
        input_i = batch % 4;
        index_in = input_slice*params.strides[0] + channel*params.strides[1];
    } else if (params.channel_dim == 1) {
        input_slice = slice;
        input_i = channel % 4;
        index_in = batch*params.strides[0] + input_slice*params.strides[1];
    } else {
        index_in = batch*params.strides[0] + channel*params.strides[1];
    }
    for(int i=params.dim_count-1; i>=2; --i) {
        int axis_size = size % params.output_sizes[i];
        if (i == params.channel_dim) {
            input_slice = axis_size / 4;
            input_i     = axis_size % 4;
            index_in += input_slice * params.strides[i];
        } else {
            index_in += axis_size * params.strides[i];
        }
        size = size / params.output_sizes[i];
    }

    ftype4 val = ftype4(
        src[index_in[0]][input_i[0]],
        src[index_in[1]][input_i[1]],
        src[index_in[2]][input_i[2]],
        src[index_in[3]][input_i[3]]
    );
    val = select(ftype4(0), val, valid_position);

    dst[index_out] = val;
}