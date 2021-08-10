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

kernel void grid_sample(const device ftype4 *input_data [[buffer(0)]],
                        const device ftype4 *x_y [[buffer(1)]],
                        device ftype4 *out [[buffer(2)]],
                        constant MetalParams &params [[buffer(3)]],
                        uint3 gid [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
                        return;


    int index =  (int)gid.z * params.output_slice * params.output_size +
                 (int)gid.y * params.output_size + (int)gid.x;

    int index_xy = (int)gid.z  * ((params.output_size-1)/(4*params.output_width) +1)*2*params.output_width +
                       (int)gid.x/(4*params.output_width) *2*params.output_width+
                        (int)gid.x%params.output_width*2;

    int in_index_xy = gid.x/params.output_width%4;



    ftype x = x_y[index_xy][in_index_xy];
    ftype y = x_y[index_xy+1][in_index_xy];

    float ix = (x + 1) * params.input_width * 0.5 - 0.5;
    float iy = (y + 1) * params.input_height * 0.5 - 0.5;

    int ix_nw = floor(ix);
    int iy_nw = floor(iy);

    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;

    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;

    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    float nw = (iy_nw >= 0 && iy_nw < params.input_height && ix_nw >= 0 && ix_nw < params.input_width) ?
               (ix_se - ix) * (iy_se - iy) : 0;
    float ne = (iy_ne >= 0 && iy_ne < params.input_height && ix_ne >= 0 && ix_ne < params.input_width) ?
               (ix - ix_sw) * (iy_sw - iy) : 0;
    float sw = (iy_sw >= 0 && iy_sw < params.input_height && ix_sw >= 0 && ix_sw < params.input_width) ?
               (ix_ne - ix) * (iy - iy_ne) : 0;
    float se = (iy_se >= 0 && iy_se < params.input_height && ix_se >= 0 && ix_se < params.input_width) ?
               (ix - ix_nw) * (iy - iy_nw) : 0;

    int nw_index = (iy_nw >= 0 && iy_nw < params.input_height && ix_nw >= 0 && ix_nw < params.input_width) ?
                    iy_nw * params.input_width + ix_nw : 0;
    int ne_index = (iy_ne >= 0 && iy_ne < params.input_height && ix_ne >= 0 && ix_ne < params.input_width) ?
                    iy_ne * params.input_width + ix_ne : 0;
    int sw_index = (iy_sw >= 0 && iy_sw < params.input_height && ix_sw >= 0 && ix_sw < params.input_width) ?
                    iy_sw * params.input_width + ix_sw : 0;
    int se_index = (iy_se >= 0 && iy_se < params.input_height && ix_se >= 0 && ix_se < params.input_width) ?
                    iy_se * params.input_width + ix_se : 0;

    int slice_index = (int)gid.z * params.input_slice * params.input_size + (int)gid.y * params.input_size;

    ftype4 val = ftype4(0,0,0,0);

    val = input_data[nw_index+slice_index] * nw + input_data[ne_index+slice_index] * ne +
             input_data[sw_index+slice_index] * sw+input_data[se_index+slice_index] * se;
    if(!isnan(x))
        out[index] = val;
    else
        out[index] = ftype4(0,0,0,0);
}



