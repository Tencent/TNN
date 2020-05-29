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
kernel void splitv_axis_1_common(const device ftype4 *src                       [[buffer(0)]],
                                                       device ftype4 *dst                            [[buffer(1)]],
                                                       constant MetalParams &params     [[buffer(2)]],
                                                       constant int &channel_offset           [[buffer(3)]],
                                                       constant int &output_slice               [[buffer(4)]],
                                                       uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, output_slice, params.batch)))
        return;
    
    int index_out = (int)gid.z*output_slice*params.output_size + (int)gid.y*params.output_size + (int)gid.x;
    
    const int input_channel_count = params.input_slice*4;
    int4 input_batch = int4(gid.z);
    int4 input_channeles = (int)gid.y*4 + int4(0, 1, 2, 3) + channel_offset;
    int4 input_x = int4(gid.x);
    int4 input_slice = input_channeles / 4;
    input_slice = min(input_slice, params.input_slice-1);
    int4 input_i = input_channeles % 4;
    
    int4 index_in = input_batch * params.input_slice * params.input_size + input_slice * params.input_size + input_x;
    
    if ( all( index_in == index_in.yzwx) &&
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
//    else {
//        auto temp = ftype4(0);
//        temp.x = src[index_in[0]][input_i[0]];
//        if (input_channeles[1] < input_channel_count) {
//            temp.y = src[index_in[1]][input_i[1]];
//        }
//        if (input_channeles[2] < input_channel_count) {
//            temp.y = src[index_in[2]][input_i[2]];
//        }
//        if (input_channeles[3] < input_channel_count) {
//            temp.y = src[index_in[3]][input_i[3]];
//        }
//        dst[index_out] = temp;
//    }
}
