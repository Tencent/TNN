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
kernel void concat_axis_1_common(const device ftype4 *src0               [[buffer(0)]],
                                 const device ftype4 *src1               [[buffer(1)]],
                                 device ftype4 *dst                      [[buffer(2)]],
                                 constant MetalConcatParams &params      [[buffer(3)]],
                                 uint3 gid                                [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    const int input_channel_0 = params.input_channel_0;
    //const int input_channel_1 = params.input_channel_1;

    int index_out = (int)gid.z*params.output_slice*params.output_size + (int)gid.y*params.output_size + (int)gid.x;
    
    int output_slice = gid.y;
    int output_channel = gid.y*4;
    if (output_channel + 4 <= input_channel_0) {
        int input_slice_0 = output_slice;
        int index_in_0 = (int)gid.z*params.input_slice_0*params.input_size + (int)input_slice_0*params.input_size + (int)gid.x;
        dst[index_out] = src0[index_in_0];
    } else if (output_channel < input_channel_0 && output_channel + 4 > input_channel_0) {
        int input_slice_0 = params.input_slice_0 - 1;
        int index_in_0 = (int)gid.z*params.input_slice_0*params.input_size + (int)input_slice_0*params.input_size + (int)gid.x;
        auto data_0 = src0[index_in_0];
        int remain_0 = input_channel_0 - output_channel;
        
        int input_slice_1 = 0;
        int index_in_1 = (int)gid.z*params.input_slice_1*params.input_size + (int)input_slice_1*params.input_size + (int)gid.x;
        auto data_1 = src1[index_in_1];
        
        if (remain_0 == 1) {
            dst[index_out] = ftype4(data_0.x, data_1.xyz);
        } else if (remain_0 == 2) {
            dst[index_out] = ftype4(data_0.xy, data_1.xy);
        } else {
            dst[index_out] = ftype4(data_0.xyz, data_1.x);
        }
    } else if (output_channel >= input_channel_0){
        int remain_1_high = (output_channel - input_channel_0) % 4;
        if (remain_1_high == 0) {
            int input_slice_1 = (output_channel - input_channel_0)/4;
            int index_in_1 = (int)gid.z*params.input_slice_1*params.input_size + (int)input_slice_1*params.input_size + (int)gid.x;
            dst[index_out] = src1[index_in_1];
        } else {
            int input_slice_1_low = (output_channel - input_channel_0)/4;
            int index_in_1_low = (int)gid.z*params.input_slice_1*params.input_size + (int)input_slice_1_low*params.input_size + (int)gid.x;
            auto data_1_low = src1[index_in_1_low];
            auto data_1_high = data_1_low;
            if (input_slice_1_low + 1 < params.input_slice_1) {
                data_1_high = src1[index_in_1_low + params.input_size];
            }
            
            if (remain_1_high == 1) {
                dst[index_out] = ftype4(data_1_low.yzw, data_1_high.x);
            } else if (remain_1_high == 2) {
                dst[index_out] = ftype4(data_1_low.zw, data_1_high.xy);
            } else {
                dst[index_out] = ftype4(data_1_low.w, data_1_high.xyz);
            }
        }
    }
}

kernel void concat_axis_1_common_x(const device ftype4 *src                              [[buffer(0)]],
                                                            device ftype *dst                                        [[buffer(1)]],
                                                            constant MetalConcatParams &params      [[buffer(2)]],
                                                            uint3 gid                                                       [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.input_size, params.input_slice_0, params.batch)))
        return;
    
    int index_in = (int)gid.z*params.input_slice_0*params.input_size + (int)gid.y*params.input_size + (int)gid.x;
    auto data_in = src[index_in];
    
    int4 input_channeles = (int)gid.y*4 + int4(0, 1, 2, 3);
    int4 output_channeles = input_channeles + params.input_channel_offset;
    int4 output_slice = output_channeles / 4;
    int4 output_i = output_channeles % 4;
    
    int4 index_out = (int)gid.z*params.output_slice*params.output_size + output_slice*params.output_size + (int)gid.x;
   
    if ( all( index_out == index_out.yzwx) &&
        all( output_i == int4(0, 1, 2, 3)) &&
        all( input_channeles < int4(params.input_channel_0)) ) {
        auto dst4 = (device ftype4 *)(dst);
        dst4[index_out.x] = data_in;
    } else  {
        dst[index_out.x*4 + output_i.x] = data_in.x;
        if (input_channeles[1] < params.input_channel_0) {
            dst[index_out.y*4 + output_i.y] = data_in.y;
        }
        if (input_channeles[2] < params.input_channel_0) {
            dst[index_out.z*4 + output_i.z] = data_in.z;
        }
        if (input_channeles[3] < params.input_channel_0) {
            dst[index_out.w*4 + output_i.w] = data_in.w;
        }
    }
    
    //below will result error
//    if ( all( output_i == int4(0, 1, 2, 3)) && all( input_channeles < int4(params.input_channel_0)) ) {
//        dst[index_out.x] = data_in;
//    } else {
//        dst[index_out.x][output_i.x] = data_in.x;
//        if (input_channeles[1] < params.input_channel_0) {
//            dst[index_out.y][output_i.y] = data_in.y;
//        }
//        if (input_channeles[2] < params.input_channel_0) {
//            dst[index_out.z][output_i.z] = data_in.z;
//        }
//        if (input_channeles[3] < params.input_channel_0) {
//            dst[index_out.w][output_i.w] = data_in.w;
//        }
//    }
}

kernel void concat_axis_23_common_x(const device ftype4 *src                          [[buffer(0)]],
                                                            device ftype4 *dst                                        [[buffer(1)]],
                                                            constant MetalConcatParams &params      [[buffer(2)]],
                                                            constant int2 &offset_xy                              [[buffer(3)]],
                                                            uint3 gid                                                       [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.input_width, params.input_height,
                         params.input_slice_0*params.batch)))
        return;

    int index_in = (int)gid.z*params.input_size + (int)gid.y*params.input_width + (int)gid.x;
    auto data_in = src[index_in];

    int index_out = (int)gid.z*params.output_size + ((int)gid.y + offset_xy.y)*params.output_width + (int)gid.x + offset_xy.x;
    dst[index_out] = data_in;
}

kernel void concat_common(const device ftype4 *src                          [[buffer(0)]],
                                device ftype4 *dst                                        [[buffer(1)]],
                                constant MetalConcatParamV2 &params      [[buffer(2)]],
                                constant int &axis_offset                           [[buffer(3)]],
                                constant int &axis_size                         [[buffer(4)]],
                                uint3 gid                                                       [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.inner_size, axis_size, params.outer_size)))
        return;

    int index_in = (int)gid.z*axis_size*params.inner_size + (int)gid.y*params.inner_size + (int)gid.x;

    int output_axis_offset = axis_offset + (int)gid.y;    
    int index_out = (int)gid.z*params.axis_size*params.inner_size + output_axis_offset*params.inner_size + (int)gid.x;

    dst[index_out] = src[index_in];
}

kernel void concat_axis_1(const device ftype4 *src                          [[buffer(0)]],
                                device ftype4 *dst                                        [[buffer(1)]],
                                constant MetalConcatParamV2 &params      [[buffer(2)]],
                                constant int &axis_offset                           [[buffer(3)]],
                                constant int &input_slice                         [[buffer(4)]],
                                constant int &input_channel                         [[buffer(5)]],
                                uint3 gid                                                       [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.inner_size, input_slice, params.outer_size)))
        return;

    int index_in = (int)gid.z*input_slice*params.inner_size + (int)gid.y*params.inner_size + (int)gid.x;
    int4 input_channel_offset = (int)gid.y*4 + int4(0, 1, 2, 3);
    bool4 valid = input_channel_offset < input_channel;
    int4 channel_offset = input_channel_offset + axis_offset;

    int4 output_slices = channel_offset / 4;
    int4 output_i = channel_offset % 4;
    int4 index_out = (int)gid.z*params.axis_size*params.inner_size + output_slices*params.inner_size + (int)gid.x;

    ftype4 val = src[index_in];

    if (axis_offset % 4 == 0 && all(valid == bool4(true))) {
        dst[index_out.x] = val;
    } else {
        auto dst1 = (device ftype*)dst;
        dst1[index_out.x * 4 + output_i.x] = val.x;
        if (valid.y == true) dst1[index_out.y * 4 + output_i.y] = val.y;
        if (valid.z == true) dst1[index_out.z * 4 + output_i.z] = val.z;
        if (valid.w == true) dst1[index_out.w * 4 + output_i.w] = val.w;
    }
}