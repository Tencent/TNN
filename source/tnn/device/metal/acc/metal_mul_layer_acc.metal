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
kernel void mul_normal(const device ftype4 *src0                                [[buffer(0)]],
                                       const device ftype4 *src1                                [[buffer(1)]],
                                       device ftype4 *dst                                            [[buffer(2)]],
                                       constant MetalBroadcastParams& params     [[buffer(3)]],
                                       uint3 gid                                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    int index_in = (int)gid.z * params.input_slice * params.input_size  + (int)gid.y * params.input_size + (int)gid.x;
    int index_out = (int)gid.z * params.output_slice * params.output_size + (int)gid.y * params.output_size + (int)gid.x;
    
    dst[index_out] = src0[index_in] * src1[index_in];
}

kernel void mul_broadcast(const device ftype4 *src0                                [[buffer(0)]],
                                           const device ftype4 *src1                                [[buffer(1)]],
                                           device ftype4 *dst                                            [[buffer(2)]],
                                           constant MetalBroadcastParams& params     [[buffer(3)]],
                                           uint3 gid                                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    const int index_size = (int)gid.y * params.input_size + (int)gid.x;
    const int index = (int)gid.z * params.input_slice * params.input_size  + index_size;
    const int batch_offset0 = params.weight_index == 0? 0 : (int)gid.z * params.input0_size;
    const int batch_offset1 = params.weight_index == 1? 0 : (int)gid.z * params.input1_size;
    
    ftype4 data0;
    if (params.broadcast_input0 == kBroadcastTypeChannel) {
        data0 = src0[batch_offset0 + gid.y];
    } else if (params.broadcast_input0 == kBroadcastTypeSingle) {
        data0 = ftype4(src0[batch_offset0 + 0].x);
    } else if (params.broadcast_input0 == kBroadcastTypeHeightWidth) {
        data0 = ftype4(src0[batch_offset0 + gid.x].x);
    } else if (params.broadcast_input0 == kBroadcastTypeElement) {
        data0 = ftype4(src0[index_size]);
    } else if (params.broadcast_input0 == kBroadcastTypeGeneral) {  //maybe some error!
        data0 = ftype4(src0[(int)(gid.x / params.output_width)]);
    } else if (params.broadcast_input0 == kBroadcastTypeChannelWidth) {
        data0 = ftype4(src0[(int)(gid.x % params.output_width) + (int)(gid.y * params.input0_size)]);
    } else if (params.broadcast_input0 == kBroadcastType5DimsHeightWidth) {
        data0 = ftype4(src0[(int)(gid.x % params.real_input0_4) + (int)(gid.y * params.real_input0_4 * params.real_input0_3)]);
    } else if (params.broadcast_input0 == kBroadcastTypeWidth) {  //kBroadcastTypeChannelHeight
        const int w = gid.x % params.input_width;
        data0 = ftype4(src0[batch_offset0 + w].x);
    } else {
        data0 = src0[index];
    }
        
    ftype4 data1;
    if (params.broadcast_input1 == kBroadcastTypeChannel) {
        data1 = src1[batch_offset1 + gid.y];
    } else if (params.broadcast_input1 == kBroadcastTypeSingle) {
        data1 = ftype4(src1[batch_offset1 + 0].x);
    } else if (params.broadcast_input1 == kBroadcastTypeHeightWidth) {
        data1 = ftype4(src1[batch_offset1 + gid.x].x);
    } else if (params.broadcast_input1 == kBroadcastTypeElement) {
        data1 = ftype4(src1[index_size]);
    } else if (params.broadcast_input1 == kBroadcastTypeGeneral) {  //maybe some error!
        data1 = ftype4(src1[(int)(gid.x / params.output_width)]);
    } else if (params.broadcast_input1 == kBroadcastTypeChannelWidth) {
        data1 = ftype4(src1[(int)(gid.x % params.output_width) + (int)(gid.y * params.input1_size)]);
    } else if (params.broadcast_input1 == kBroadcastType5DimsHeightWidth) {
        data1 = ftype4(src1[(int)(gid.x % params.real_input1_4) + (int)(gid.y * params.real_input1_4 * params.real_input1_3)]);
    } else if (params.broadcast_input1 == kBroadcastTypeWidth) {   //kBroadcastTypeChannelHeight
        const int w = gid.x % params.input_width;
        data1 = ftype4(src1[batch_offset1 + w].x);
    } else {
        data1 = src1[index];
    }
    
    int index_out = (int)gid.z * params.output_slice * params.output_size + (int)gid.y * params.output_size + (int)gid.x;
    dst[index_out] = data0 * data1;
}
