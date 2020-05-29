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

kernel void convolution_common(const device ftype4 *in            [[buffer(0)]],
                                     device ftype4 *out                 [[buffer(1)]],
                                     constant MetalConvParams& params   [[buffer(2)]],
                                     const device ftype4x4 *wt          [[buffer(3)]],
                                     const device ftype4 *biasTerms     [[buffer(4)]],
                                     uint3 gid                          [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width,
                           params.output_height,
                           params.output_slice)))
        return;
    
    int offset_x = (int)gid.x * params.stride_x - params.pad_x;
    int offset_y = (int)gid.y * params.stride_y - params.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, params.dilation_x)));
    int ex = min(params.kernel_x, UP_DIV(params.input_width - offset_x, params.dilation_x));
    short kw = ex - sx;
    int sy = max(0, (UP_DIV(-offset_y, params.dilation_y)));
    int ey = min(params.kernel_y, UP_DIV(params.input_height - offset_y, params.dilation_y));
    short kh = ey - sy;
    offset_x += sx * params.dilation_x;
    offset_y += sy * params.dilation_y;
    
    auto z_in  = in                                                   + offset_y * params.input_width    + offset_x;
    auto z_wt  = wt  + (int)gid.z * params.input_slice * params.kernel_size + sy * params.kernel_x             + sx;
    auto z_out = out + (int)gid.z * params.output_size                   + (int)gid.y * params.output_width + (int)gid.x;
    
    int dilation_h = params.input_width * params.dilation_y;
    auto result = params.has_bias ? float4(biasTerms[gid.z]) : float4(Zero4);
    for (auto z = 0; z < params.input_slice; z++) {
        for (auto y = 0; y < kh; y++) {
            for (auto x = 0; x < kw; x++) {
                auto wt4 = float4x4(z_wt[z * params.kernel_size + y * params.kernel_x + x]);
                auto in4 = float4(z_in[z * params.input_size  + y * dilation_h   + x * params.dilation_x]);
                result += in4 * wt4;
            }
        }
    }
    
    *z_out = activate(ftype4(result), params.activation);
}
