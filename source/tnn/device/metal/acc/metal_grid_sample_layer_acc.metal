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


kernel void grid_sample(device ftype4 *img [[buffer(0)]],
                        device ftype4 *x_y [[buffer(1)]],
                        device ftype4 *out [[buffer(2)]],
                        constant MetalParams &params [[buffer(3)]],
                        uint3 gid [[thread_position_in_grid]]) {
                        return;
     if (any(gid >= uint3(params.output_size, params.output_slice, params.batch)))
                        return;
/*
    int index =  (int)gid.z * params.output_slice * params.output_size +
                 (int)gid.y * params.output_size + (int)gid.x;
    ftype x = x_y[index][0];
    ftype y = x_y[index][1];
    int x0 = int(x);
    int y0 = int(y);
    int x1 = int(x+1);
    int y1 = int(y+1);
    ftype4 val = (x1-x)*(y1-y)/((x1-x0)*(y1-y0))*img[y0*params.input_width+x0]+
         (x-x0)*(y1-y)/((x1-x0)*(y1-y0))*img[y0*params.input_width+x1]+
         (x1-x)*(y-y0)/((x1-x0)*(y1-y0))*img[y1*params.input_width+x0]+
         (x-x0)*(y-y0)/((x1-x0)*(y1-y0))*img[y1*params.input_width+x1]+1;


    out[index] = val;
    */
}



