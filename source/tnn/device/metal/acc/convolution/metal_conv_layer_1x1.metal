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

#define CONV_UNROLL (4)

kernel void convolution_1x1_4x(const device ftype4 *in           [[buffer(0)]],
                            device ftype4 *out                [[buffer(1)]],
                            constant MetalConvParams& params  [[buffer(2)]],
                            const device ftype4x4 *wt         [[buffer(3)]],
                            const device ftype4 *biasTerms    [[buffer(4)]],
                            uint3 gid                         [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_size,
                           params.output_slice,
                           params.batch)))
        return;
    
    int g = gid.y / params.output_slice_per_group;
    auto xy_wt  = wt  + (int)gid.y * params.input_slice_per_group;
    auto xy_in  = in  + (int)gid.z * params.input_slice  * params.input_size  + g * params.input_slice_per_group * params.input_size  + (int)gid.x;
    auto xy_out = out + (int)gid.z * params.output_slice * params.output_size + (int)gid.y * params.output_size + (int)gid.x;
    
    auto result = params.has_bias ? float4(biasTerms[gid.y]) : float4(Zero4);
    for (auto z = 0; z < params.input_slice_per_group; z++, xy_in += params.input_size) {
        result += float4(*xy_in) * float4x4(xy_wt[z]);
    }
    *xy_out = activate(ftype4(result), params.activation);
}

kernel void convolution_1x1_common(const device ftype *in     [[buffer(0)]],
                            device ftype *out                [[buffer(1)]],
                            constant MetalConvParams& params  [[buffer(2)]],
                            const device ftype4 *wt           [[buffer(3)]],
                            const device ftype *biasTerms    [[buffer(4)]],
                            uint3 gid                         [[thread_position_in_grid]]) {
    uint output_slice_per_group = UP_DIV(params.output_slice_per_group, 4);
    uint slices = params.group * output_slice_per_group;
    if (any(gid >= uint3(params.output_size,
                           slices,
                           params.batch)))
        return;

    int g = gid.y / output_slice_per_group;
    int4 output_channel = (gid.y % output_slice_per_group) * 4 + int4(0, 1, 2, 3);
    bool4 valid = output_channel < params.output_slice_per_group;
    output_channel += g * params.output_slice_per_group;
    
    auto output_slice = output_channel / 4;
    auto output_c     = output_channel % 4;

    int start_input_channel = g * params.input_slice_per_group;
    int start_input_slice   = start_input_channel / 4;
    int start_input_c       = start_input_channel % 4;

    const device ftype4 *xy_wt  = wt  + (int)gid.y * params.input_slice_per_group;
    const device ftype *xy_in   = in  + (int)gid.z * params.input_slice  * params.input_size * 4 + \
                    start_input_slice * params.input_size * 4 + (int)gid.x * 4 + start_input_c;

    int4 out_idx = (int)gid.z * params.output_slice * params.output_size * 4 + \
                    output_slice * params.output_size * 4 + (int)gid.x * 4 + output_c;

    auto bias = params.has_bias ? (select(float4(Zero4),
                        float4(biasTerms[output_channel[0]],
                                biasTerms[output_channel[1]],
                                biasTerms[output_channel[2]],
                                biasTerms[output_channel[3]]),
                        valid)) : float4(Zero4);
    
    float4 sum = bias;
    const int step_size = params.input_size * 4 - 3;
    for (auto c = 0; c < params.input_slice_per_group; c++) {
        sum += float(*xy_in) * float4(xy_wt[c]);
        start_input_c += 1;
        int flag = start_input_c == 4;
        xy_in += flag * step_size + (1 - flag) * 1;
        start_input_c -= flag * start_input_c;
    }
    ftype4 result = activate(ftype4(sum), params.activation);
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

kernel void convolution_1x1_g1z4(const device ftype4 *in             [[buffer(0)]],
                                 device ftype4 *out                  [[buffer(1)]],
                                 constant MetalConvParams& params    [[buffer(2)]],
                                 const device ftype4x4 *wt           [[buffer(3)]],
                                 const device ftype4 *biasTerms      [[buffer(4)]],
                                 uint3 gid                           [[thread_position_in_grid]]) {
    if ((int)gid.x >= params.output_size || (int)gid.y * CONV_UNROLL >= params.output_slice || (int)gid.z >= params.batch) return;
    
    int uz = gid.y * CONV_UNROLL;
    auto xy_wt0 = wt + uz * params.input_slice;
    auto xy_wt1 = uz + 1 < params.output_slice ? xy_wt0 + params.input_slice : nullptr;
    auto xy_wt2 = uz + 2 < params.output_slice ? xy_wt1 + params.input_slice : nullptr;
    auto xy_wt3 = uz + 3 < params.output_slice ? xy_wt2 + params.input_slice : nullptr;
    auto xy_in  = in  + (int)gid.z * params.input_slice  * params.input_size                         + (int)gid.x;
    auto xy_out = out + (int)gid.z * params.output_slice * params.output_size + uz * params.output_size + (int)gid.x;
    
    float4 result0 = 0, result1 = 0, result2 = 0, result3 = 0;
    for (auto z = 0; z < params.input_slice; z++, xy_in += params.input_size) {
        auto in4 = float4(*xy_in);
        /* true */  result0 += in4 * float4x4(xy_wt0[z]);
        if (xy_wt1) result1 += in4 * float4x4(xy_wt1[z]);
        if (xy_wt2) result2 += in4 * float4x4(xy_wt2[z]);
        if (xy_wt3) result3 += in4 * float4x4(xy_wt3[z]);
    }
    
    
    if (params.has_bias) {
        *xy_out = activate(ftype4(result0 + float4(biasTerms[uz + 0])), params.activation);
        if (xy_wt1) { xy_out += params.output_size; *xy_out = activate(ftype4(result1 + float4(biasTerms[uz + 2])), params.activation); }
        if (xy_wt2) { xy_out += params.output_size; *xy_out = activate(ftype4(result2 + float4(biasTerms[uz + 2])), params.activation); }
        if (xy_wt3) { xy_out += params.output_size; *xy_out = activate(ftype4(result3 + float4(biasTerms[uz + 3])), params.activation); }
    } else {
        *xy_out = activate(ftype4(result0), params.activation);
        if (xy_wt1) { xy_out += params.output_size; *xy_out = activate(ftype4(result1), params.activation); }
        if (xy_wt2) { xy_out += params.output_size; *xy_out = activate(ftype4(result2), params.activation); }
        if (xy_wt3) { xy_out += params.output_size; *xy_out = activate(ftype4(result3), params.activation); }
    }
}

