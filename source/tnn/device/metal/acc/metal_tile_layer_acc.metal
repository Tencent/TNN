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
/*
kernel void tile(device ftype4 *in [[buffer(0)]],
                device ftype4 *out [[buffer(1)]],
                constant MetalTileParams &params [[buffer(2)]],
                uint3 gid [[thread_position_in_grid]]) {
     if (any(gid >= uint3(params.output_size, params.output_channel, params.batch)))
                        return;
    int plane_row = params.extend_width_times * params.input_width;
    int plane = plane_row * params.input_height * params.extend_height_times;
    int in_plane = params.input_width * params.input_height;
    //auto z_out = out + (int)gid.z * params.output_slice * params.output_size +
    //                 (int)gid.y * params.output_size + (int)gid.x;
    int o_index =  (int)gid.z * params.output_slice * params.output_size +
                         (int)gid.y /4 * params.output_size  + (int)gid.x;  //not pad has problem

    //auto z_in = in + (int)gid.z / params.extend_batch_times * params.input_slice * params.input_size +
    //                (int)gid.y / params.extend_slice_times * params.input_size +
    //                (int)gid.x % plane / plane_row * params.input_width +
    //                (int)gid.x % plane % params.input_width;
    //int index = (int)gid.z % params.batch * params.input_slice  * params.input_size +
    //                                (int)gid.y % params.input_slice *params.input_size +
    //                                (int)gid.x % (4*plane) / plane * in_plane +
    //                                (int)gid.x % plane / plane_row * params.input_width +
    //                                (int)gid.x % plane % params.input_width;

    int offset = (int)gid.z % params.batch * params.input_slice * params.input_size +
                     (int)gid.y % params.input_channel +
                     (int)gid.x % plane / plane_row * params.input_width +
                     (int)gid.x % plane % params.input_width;

    if(params.output_slice == 1)
    {
        for(int i=0; i < params.extend_channel_times; i++)
        {
            out[o_index][i*params.input_channel] = in[offset][0];
        }
    }
    else
    {
        out[o_index][(int)gid.y%4] = in[offset][0];
    }
}

*/
/*
kernel void tile(const device ftype4 *in [[buffer(0)]],
                device ftype4 *out [[buffer(1)]],
                constant MetalTileParams &params [[buffer(2)]],
                uint3 gid [[thread_position_in_grid]]) {
     if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
                        return;
    int plane_row = params.extend_width_times * params.input_width;
    int plane = plane_row * params.input_height;
    auto z_out = out + (int)gid.z * params.output_slice * params.output_size +
                     (int)gid.y * params.output_size + (int)gid.x;

    //auto z_in = in + (int)gid.z / params.extend_batch_times * params.input_slice * params.input_size +
    //                (int)gid.y / params.extend_slice_times * params.input_size +
    //                (int)gid.x % plane / plane_row * params.input_width +
    //                (int)gid.x % plane % params.input_width;

    auto z_in = in + (int)gid.z / params.extend_batch_times * params.input_slice * params.input_size +
                     (int)gid.y / params.extend_slice_times * params.input_size +
                     (int)gid.x % plane / plane_row * params.input_width +
                     (int)gid.x % plane % params.input_width;

    *z_out = *z_in;
}
*/
kernel void tile(const device ftype4 *in [[buffer(0)]],
                device ftype4 *out [[buffer(1)]],
                constant MetalTileParams &params [[buffer(2)]],
                uint3 gid [[thread_position_in_grid]]) {
     if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
                        return;
    int plane_row = params.extend_width_times * params.input_width;
    int plane = plane_row * params.input_height;
    auto z_out = out + (int)gid.z * params.output_slice * params.output_size +
                     (int)gid.y * params.output_size + (int)gid.x;

    int batch_index = (int)gid.z % (params.batch/params.extend_batch_times) * params.input_slice * params.input_size;
    int x_index =(int)gid.x % plane / plane_row * params.input_width +
                                      (int)gid.x % params.input_width;
    int start_row = (int)gid.y * 4 - (int)gid.y * 4 / params.input_channel * params.input_channel;
    ftype a[4];
    int count = 0;
    for(int i = 0; i < 4; i++)
    {
        if(start_row >= params.input_channel)
        {
            start_row = 0;
            count++;
        }
        if(count > params.extend_channel_times)
            a[i] = 0;
        else
        {
                a[i] = in[batch_index + start_row/4*params.input_size + x_index][start_row%4];
        }
        start_row++;
    }
    ftype4 tmp = ftype4(a[0],a[1],a[2],a[3]);
    *z_out = tmp;
}