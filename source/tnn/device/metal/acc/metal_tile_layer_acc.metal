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
    int in_channel_offset =(int)gid.x % plane / plane_row * params.input_width +
                                          (int)gid.x % params.input_width;
    int channel_index = (int)gid.y*4  - (int)gid.y * 4 / params.input_channel * params.input_channel;

    ftype a[4];
    int count = 0;
    for(int i = 0; i < 4; i++)
    {
        if(channel_index >= params.input_channel)    //channel_index must in range(0,input_channel)
        {
            channel_index = 0;
            count++;                                //if channel_index==input_channel, channel_index re-assign to 0,count is this case happen times in one slice
        }
        if(count > params.extend_channel_times)
            a[i] = 0;                               //after channel extended, output channel < 4,need pad 0
        else
            // channel_index/4 and channel_index%4 means four channel element packed as one ftype4 element
            a[i] = in[batch_index + channel_index/4*params.input_size + in_channel_offset][channel_index%4];

        channel_index++;
    }
    ftype4 tmp = ftype4(a[0],a[1],a[2],a[3]);
    *z_out = tmp;
}