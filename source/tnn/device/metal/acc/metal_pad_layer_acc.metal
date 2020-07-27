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
kernel void pad_const_common(const device ftype4 *src                  [[buffer(0)]],
                                                     device ftype4 *dst                            [[buffer(1)]],
                                                     constant MetalPadParams &params     [[buffer(2)]],
                                                     uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch)))
        return;
    
    int index_out = (int)gid.z*params.output_size + (int)gid.y*params.output_width + (int)gid.x;
    int index_in_y = (int)gid.y - params.pad_t ;
    int index_in_x = (int)gid.x - params.pad_l;

    int index_out_c_beg = ((int)gid.z % params.output_slice ) * 4;
    int index_out_c_end = index_out_c_beg + 4;
    
    auto temp = ftype4(params.value);
    if (index_in_y >= 0 && index_in_y < params.input_height &&
        index_in_x >= 0 && index_in_x < params.input_width) {
            int num_pad_c_beg = max(params.pad_c_b - index_out_c_beg, 0);
            int num_pad_c_end = max(index_out_c_end - (params.pad_c_b + params.input_channel), 0);
            int num_from_input = 4 - num_pad_c_beg - num_pad_c_end;
            // we need to fill temp with some input elements
            if(num_from_input > 0){
                int index_in_channel = index_out_c_beg < params.pad_c_b? 0 : index_out_c_beg - params.pad_c_b;
                int index_in_batch = (int)gid.z / params.output_slice;

                for(int i=0; i<num_from_input; ++i){
                    int index_in_slice = index_in_channel / 4;
                    int index_in_c = index_in_channel%4;

                    int index_in = (index_in_batch*params.input_slice*params.input_size) + index_in_slice*params.input_size + index_in_y*params.input_width + index_in_x;
                    temp[num_pad_c_beg + i] = src[index_in][index_in_c];
                    index_in_channel += 1;
                }
            }
    }
    dst[index_out] =  temp;
}

//specialization for performing padding at channel dimension when pad_c_b is multiple of 4
kernel void pad_const_channel4(const device ftype4 *src                  [[buffer(0)]],
                                     device ftype4 *dst                  [[buffer(1)]],
                                     constant MetalPadParams &params     [[buffer(2)]],
                                     uint3 gid                           [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch)))
        return;
    
    int index_out = (int)gid.z*params.output_size + (int)gid.y*params.output_width + (int)gid.x;
    int index_in_y = (int)gid.y - params.pad_t ;
    int index_in_x = (int)gid.x - params.pad_l;
    // the param.pad_c_b must be multiple of 4
    int index_in_slice = (int)gid.z % params.output_slice - params.pad_c_b / 4;

    auto temp = ftype4(params.value);
    if (index_in_y >= 0 && index_in_y < params.input_height &&
        index_in_x >= 0 && index_in_x < params.input_width  &&
        index_in_slice >=0  && index_in_slice < params.input_slice) {
            int index_in_batch = (int)gid.z / params.output_slice;
            int index_in = index_in_batch*params.input_slice*params.input_size + index_in_slice*params.input_size + index_in_y*params.input_width + index_in_x;
            temp = src[index_in];
    }
    dst[index_out] =  temp;
}

kernel void pad_reflect_common(const device ftype4 *src                  [[buffer(0)]],
                                                     device ftype4 *dst                            [[buffer(1)]],
                                                     constant MetalPadParams &params     [[buffer(2)]],
                                                     uint3 gid                                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch)))
        return;
    
    int index_out = (int)gid.z*params.output_size + (int)gid.y*params.output_width + (int)gid.x;
    int index_in_y = (int)gid.y - params.pad_t ;
    int index_in_x = (int)gid.x - params.pad_l;
    
    if (index_in_y < 0) {
        index_in_y = - index_in_y;
    } else if (index_in_y >= params.input_height){
        index_in_y = params.input_height - (index_in_y - params.input_height) - 2;
    }
    
    if (index_in_x < 0) {
        index_in_x = - index_in_x;
    } else if (index_in_x >= params.input_width){
        index_in_x = params.input_width - (index_in_x - params.input_width) - 2;
    }
    
    int index_in = (int)gid.z*params.input_size + index_in_y * params.input_width + index_in_x;
    
    dst[index_out] =  src[index_in];
}
