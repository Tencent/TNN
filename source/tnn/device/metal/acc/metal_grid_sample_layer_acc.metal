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

kernel void grid_sample(const device ftype4 *in                 [[buffer(0)]],
                        const device ftype4 *grid               [[buffer(1)]],
                                      device ftype4 *out                      [[buffer(2)]],
                                      constant MetalParams &params            [[buffer(3)]],
                                      uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch)))
        return;

    int batch = gid.z / params.output_slice;

    auto grid_slice = gid.y / 4;
    auto grid_c     = gid.y % 4;
    auto gid_offset = ((batch * UP_DIV(params.output_height, 4) + grid_slice) * params.output_width + gid.x) * 2;
    auto grid_x = grid[gid_offset + 0][grid_c];
    auto grid_y = grid[gid_offset + 1][grid_c];
    grid_x = (grid_x + 1) * params.input_width  * 0.5 - 0.5;
    grid_y = (grid_y + 1) * params.input_height * 0.5 - 0.5;

    int left = floor(grid_x);
    int right = left < params.input_width-1 ? left + 1 : left;
    int top  = floor(grid_y);
    int bottom = top < params.input_height-1 ? top + 1 : top;

    float w1lambda = grid_x - left;
    float w0lambda = 1.f - w1lambda;
    if (left < 0 || left > params.input_width - 1) {
        w0lambda = 0;
    }
    if (left + 1 < 0 || left + 1 > params.input_width - 1) {
        w1lambda = 0;
    }
    float h1lambda = grid_y - top;
    float h0lambda = 1.f - h1lambda;
    if (top < 0 || top > params.input_height - 1) {
        h0lambda = 0;
    }
    if (top + 1 < 0 || top + 1 > params.input_height - 1) {
        h1lambda = 0;
    }

    auto in_z        = in + gid.z * params.input_size;
    auto in_top      = in_z + top * params.input_width;
    auto in_bottom   = in_z + bottom * params.input_width;
    // Note: we donot check if these 4 points are valid
    auto tl = ftype4(in_top[left])     * w0lambda * h0lambda;
    auto tr = ftype4(in_top[right])    * w1lambda * h0lambda;
    auto bl = ftype4(in_bottom[left])  * w0lambda * h1lambda;
    auto br = ftype4(in_bottom[right]) * w1lambda * h1lambda;
    out[int(gid.z) * params.output_size + int(gid.y) * params.output_width + int(gid.x)] = ftype4(tl + tr + bl + br);
}
