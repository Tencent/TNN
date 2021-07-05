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
kernel void deconv_common_group_channel_in4x_out4x(const device ftype4 *in          [[buffer(0)]],
                                                   device ftype4 *out               [[buffer(1)]],
                                                   constant MetalConvParams& param   [[buffer(2)]],
                                                   const device ftype4x4 *wt        [[buffer(3)]],
                                                   const device ftype4 *biasTerms   [[buffer(4)]],
                                                   uint3 gid                        [[thread_position_in_grid]]) {
    if ((int)gid.x >= param.output_width || (int)gid.y >= param.output_height) return;
    
    int b = gid.z / param.output_slice;
    int o = gid.z % param.output_slice;
    float4 result = param.has_bias ? float4(biasTerms[o]) : float4(Zero4);
    
    
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
        auto o_wt = wt + o * param.input_slice * param.kernel_size;
        auto b_in = in + b * param.input_slice * param.input_size;
        for (auto z = 0; z < param.input_slice; z++) {
            for (auto ky = max_ky, iy = min_iy; ky >= min_ky; ky -= param.kernel_delta_y, iy += param.input_delta_y) {
                for (auto kx = max_kx, ix = min_ix; kx >= min_kx; kx -= param.kernel_delta_x, ix += param.input_delta_x) {
                    auto wt4 = float4x4(o_wt[z * param.kernel_size + ky * param.kernel_x + kx]);
                    auto in4 = float4(b_in[z * param.input_size + iy * param.input_width + ix]);
                    result += float4(in4 * wt4);
                }
            }
        }
    }
    
    out[(int)gid.z * param.output_size + (int)gid.y * param.output_width + (int)gid.x] = activate(ftype4(result), param.activation);
}


kernel void deconv_common_group_channel_in2_out1_group2(const device ftype4 *in          [[buffer(0)]],
                                                        device ftype4 *out               [[buffer(1)]],
                                                        constant MetalConvParams& param   [[buffer(2)]],
                                                        const device ftype4x4 *wt        [[buffer(3)]],
                                                        const device ftype4 *biasTerms   [[buffer(4)]],
                                                        uint3 gid                        [[thread_position_in_grid]]) {
    if ((int)gid.x >= param.output_width || (int)gid.y >= param.output_height) return;
    
    float4 result0 = param.has_bias ? float4(biasTerms[0]) : float4(Zero4);
    float4 result1 = float4(result0.y);
    
    
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
        auto o_wt0 = wt + 0;
        auto o_wt1 = wt + param.kernel_size;
        auto b_in = in + 0;
        for (auto ky = max_ky, iy = min_iy; ky >= min_ky; ky -= param.kernel_delta_y, iy += param.input_delta_y) {
            for (auto kx = max_kx, ix = min_ix; kx >= min_kx; kx -= param.kernel_delta_x, ix += param.input_delta_x) {
                auto wt4_0 = o_wt0[ky * param.kernel_x + kx];
                auto wt4_1 = o_wt1[ky * param.kernel_x + kx];
                auto in4 = b_in[iy * param.input_width + ix];
                
                result0 += float4(float2(in4.xy), 0.0f, 0.0f) * float4x4(wt4_0);
                result1 += float4(float2(in4.zw), 0.0f, 0.0f) * float4x4(wt4_1);
            }
        }
    }
    
    out[(int)gid.y * param.output_width + (int)gid.x] = activate(ftype4(result0.x, result1.x , 0.0f, 0.0f), param.activation);
}

kernel void deconv_common_group_channel(const device ftype *in          [[buffer(0)]],
                                        device ftype *out               [[buffer(1)]],
                                        constant MetalConvParams& param   [[buffer(2)]],
                                        const device ftype4 *wt         [[buffer(3)]],
                                        const device ftype *biasTerms   [[buffer(4)]],
                                        constant int& group             [[buffer(5)]],
                                        uint3 gid                       [[thread_position_in_grid]]) {
    if (any(gid >= uint3(param.output_width,
                           param.output_height,
                           param.output_slice)))
        return;

    int4 local_channel = int(gid.z) * 4 + int4(0, 1, 2, 3);
    bool4 valid = local_channel < param.output_slice_per_group;
    auto output_channel = local_channel + group * param.output_slice_per_group;

    auto bias = param.has_bias ? (select(float4(Zero4),
                        float4(biasTerms[local_channel[0]],
                                biasTerms[local_channel[1]],
                                biasTerms[local_channel[2]],
                                biasTerms[local_channel[3]]),
                        valid)) : float4(Zero4);
    float4 sum = bias;
    
    auto output_slice = output_channel / 4;
    auto output_c     = output_channel % 4;

    int start_input_channel = group * param.input_slice_per_group;
    int start_input_slice   = start_input_channel / 4;
    int start_input_c       = start_input_channel % 4;

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

        const device ftype4 *xy_wt = wt + (int)gid.z * param.input_slice_per_group * param.kernel_size;
        const device ftype  *xy_in = in + start_input_slice * param.input_size * 4 + start_input_c;
        const auto step_size = param.input_size * 4 - 3;
        
        for (auto c = 0; c < param.input_slice_per_group; c++) {
            for (auto ky = max_ky, iy = min_iy; ky >= min_ky; ky -= param.kernel_delta_y, iy += param.input_delta_y) {
                for (auto kx = max_kx, ix = min_ix; kx >= min_kx; kx -= param.kernel_delta_x, ix += param.input_delta_x) {
                    auto wt4 = float4(xy_wt[ky * param.kernel_x + kx]);
                    auto in = float(xy_in[(iy * param.input_width + ix)*4]);
                    sum += in * wt4;
                }
            }
            start_input_c += 1;
            int flag = start_input_c == 4;
            xy_in += flag * step_size + (1 - flag) * 1;
            start_input_c -= flag * start_input_c;
            xy_wt += param.kernel_size;
        }
    }

    int4 out_idx = output_slice * param.output_size * 4 + \
                        ((int)gid.y * param.output_width + (int)gid.x) * 4 + output_c;
    ftype4 result = activate(ftype4(sum), param.activation);
    if (valid[3]) {
        out[out_idx[0]] = result[0];
        out[out_idx[1]] = result[1];
        out[out_idx[2]] = result[2];
        out[out_idx[3]] = result[3];
    } else if (valid[2]) {
        out[out_idx[0]] = result[0];
        out[out_idx[1]] = result[1];
        out[out_idx[2]] = result[2];
    } else if (valid[1]) {
        out[out_idx[0]] = result[0];
        out[out_idx[1]] = result[1];
    } else {
        out[out_idx[0]] = result[0];
    }

}
