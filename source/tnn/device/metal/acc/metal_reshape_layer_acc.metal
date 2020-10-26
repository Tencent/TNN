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
kernel void reshape_common_nchw(const device ftype4 *src                  [[buffer(0)]],
                                                device ftype4 *dst                            [[buffer(1)]],
                                                constant MetalReshapeParams &params     [[buffer(2)]],
                                                uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    int index_out = (int)gid.y*params.output_size + (int)gid.x;
    int4 index_nchw = ((int)gid.y*4 + int4(0, 1, 2, 3))*params.output_size + (int)gid.x;

    int chw_size = params.output_size * params.output_channel;
    bool4 flag = (index_nchw >= chw_size);
    
    index_out += (int)gid.z*params.output_slice*params.output_size;
    
    int4 input_batch = int4(gid.z);
    int4 input_channel = index_nchw / params.input_size;
    int4 input_x = index_nchw - input_channel * params.input_size;
    int4 input_slice = input_channel / 4;
    int4 input_i = input_channel % 4;
    
    int4 index_in = input_batch * params.input_slice * params.input_size + input_slice * params.input_size + input_x;

    ftype4 val = ftype4(0);
    if (flag[1] == true) {
        val = ftype4(
            src[index_in[0]][input_i[0]],
            0,
            0,
            0
        );
    } else if (flag[2] == true) {
        val = ftype4(
            src[index_in[0]][input_i[0]],
            src[index_in[1]][input_i[1]],
            0,
            0
        );
    } else if(flag[3] == true) {
        val = ftype4(
            src[index_in[0]][input_i[0]],
            src[index_in[1]][input_i[1]],
            src[index_in[2]][input_i[2]],
            0
        );
    } else {
        val = ftype4(
            src[index_in[0]][input_i[0]],
            src[index_in[1]][input_i[1]],
            src[index_in[2]][input_i[2]],
            src[index_in[3]][input_i[3]]
        );
    }
    dst[index_out] = val;
}

kernel void reshape_common_nhwc(const device ftype4 *src                  [[buffer(0)]],
                                                device ftype4 *dst                            [[buffer(1)]],
                                                constant MetalReshapeParams &params     [[buffer(2)]],
                                                uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;

    int index_out = (int)gid.y*params.output_size + (int)gid.x;
    int4 index_nhwc = (int)gid.y*4 + int4(0, 1, 2, 3) + (int)gid.x*params.output_channel;

    int chw_size = params.output_size * params.output_channel;
    bool4 flag = (index_nhwc >= chw_size);

    index_out += (int)gid.z*params.output_slice*params.output_size;

    int4 input_batch = int4(gid.z);
    int4 input_x = index_nhwc / (params.input_channel);
    int4 input_channel = index_nhwc - input_x * params.input_channel;
    int4 input_slice = input_channel / 4;
    int4 input_i = input_channel % 4;

    int4 index_in = input_batch * params.input_slice * params.input_size + input_slice * params.input_size + input_x;

    ftype4 val = ftype4(0);
    if (flag[1] == true) {
        val = ftype4(
            src[index_in[0]][input_i[0]],
            0,
            0,
            0
        );
    } else if (flag[2] == true) {
        val = ftype4(
            src[index_in[0]][input_i[0]],
            src[index_in[1]][input_i[1]],
            0,
            0
        );
    } else if(flag[3] == true) {
        val = ftype4(
            src[index_in[0]][input_i[0]],
            src[index_in[1]][input_i[1]],
            src[index_in[2]][input_i[2]],
            0
        );
    } else {
        val = ftype4(
            src[index_in[0]][input_i[0]],
            src[index_in[1]][input_i[1]],
            src[index_in[2]][input_i[2]],
            src[index_in[3]][input_i[3]]
        );
    }
    dst[index_out] = val;
}
