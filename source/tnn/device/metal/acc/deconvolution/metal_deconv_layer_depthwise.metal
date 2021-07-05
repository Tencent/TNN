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
kernel void deconv_depthwise(const device ftype4 *in          [[buffer(0)]],
                             device ftype4 *out               [[buffer(1)]],
                             constant MetalConvParams& param  [[buffer(2)]],
                             const device ftype4 *wt          [[buffer(3)]],
                             const device ftype4 *biasTerms   [[buffer(4)]],
                             uint3 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= param.output_width || (int)gid.y >= param.output_height) return;
    
    const int out_slice = int(gid.z) % param.output_slice;

    float4 result = param.has_bias ? float4(biasTerms[out_slice]) : float4(Zero4);
    int oy = (int)gid.y + param.pad_y;
    int ox = (int)gid.x + param.pad_x;
    int max_sy = min((param.input_height - 1) * param.stride_y, oy / param.stride_y * param.stride_y);
    int max_sx = min((param.input_width - 1) * param.stride_x, ox / param.stride_x * param.stride_x);
    int min_ky = UP_DIV(oy - max_sy, param.dilation_y);
    int min_kx = UP_DIV(ox - max_sx, param.dilation_x);
    if ((oy - min_ky * param.dilation_y) % param.stride_y == 0 && (ox - min_kx * param.dilation_x) % param.stride_x == 0) {
        
        int min_sy = max(0, ROUND_UP(oy + param.dilation_y - param.kernel_y * param.dilation_y, param.stride_y));
        int min_sx = max(0, ROUND_UP(ox + param.dilation_x - param.kernel_x * param.dilation_x, param.stride_x));
        int max_ky = (oy - min_sy) / param.dilation_y;
        int max_kx = (ox - min_sx) / param.dilation_x;
        int min_iy = (oy - max_ky * param.dilation_y) / param.stride_y;
        int min_ix = (ox - max_kx * param.dilation_x) / param.stride_x;
        auto z_wt = wt + out_slice * param.kernel_size;
        auto z_in = in + (int)gid.z * param.input_size;
        for (auto ky = max_ky, iy = min_iy; ky >= min_ky; ky -= param.kernel_delta_y, iy += param.input_delta_y) {
            for (auto kx = max_kx, ix = min_ix; kx >= min_kx; kx -= param.kernel_delta_x, ix += param.input_delta_x) {
                auto wt4 = float4(z_wt[ky * param.kernel_x + kx]);
                auto in4 = float4(z_in[iy * param.input_width + ix]);
                result += in4 * wt4;
            }
        }
    }
    out[(int)gid.z * param.output_size + (int)gid.y * param.output_width + (int)gid.x] = activate(ftype4(result), param.activation);
}


