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

template <typename IType, typename OType>
static inline void matmul4x4_template(const device IType *in,
                                      device OType *out,
                                      const device IType *kt,
                                      constant MetalMatMul4x4Params &params,
                                      uint3 gid) {
    if ((int)gid.x >= params.output_width || (int)gid.y >= params.output_height)
        return;
    
    auto ky = (int)gid.y + (int)gid.z * params.output_height;
    auto iy = (int)gid.x + (int)gid.z * params.output_width;
    auto off_in  = in  + iy * params.multi_length;
    auto off_wt  = kt  + ky * params.multi_length;
    auto off_out = out + iy + 4 * (int)gid.y * params.output_width * params.group;
    
    float4 result0 = 0, result1 = 0, result2 = 0, result3 = 0;
    for (int k = 0; k < params.multi_length; ++k) {
        auto w4x4 = float4x4(off_wt[k]);
        auto i4x4 = float4x4(off_in[k]);
        result0 += w4x4 * i4x4[0];
        result1 += w4x4 * i4x4[1];
        result2 += w4x4 * i4x4[2];
        result3 += w4x4 * i4x4[3];
    }
    *off_out = OType(result0); off_out += params.output_width * params.group;
    *off_out = OType(result1); off_out += params.output_width * params.group;
    *off_out = OType(result2); off_out += params.output_width * params.group;
    *off_out = OType(result3);
}

kernel void matmul4x4(const device ftype4x4 *in     [[buffer(0)]],
                      device ftype4 *out            [[buffer(1)]],
                      const device ftype4x4 *kt     [[buffer(2)]],
                      constant MetalMatMul4x4Params &params [[buffer(3)]],
                      uint3 gid                     [[thread_position_in_grid]]) {
    matmul4x4_template<ftype4x4, ftype4>(in, out, kt, params, gid);
}



static inline ftype4 get_input(const device ftype4 *input, int x, int y, constant MetalWinogradParams &params) {
    return x < params.input_width && y < params.input_height && x >= 0 && y >= 0 ? input[x + y * params.input_width] : Zero4;
}

kernel void winograd_transform_source2_3_1(const device ftype4 *in          [[buffer(0)]],
                                           device ftype4 *out               [[buffer(1)]],
                                           constant MetalWinogradParams &params [[buffer(2)]],
                                           uint3 gid                        [[thread_position_in_grid]]) {
    if ((int)gid.x >= params.unit_width || (int)gid.y >= params.unit_height)
        return;
    
    auto pos = int3(gid);
    
    int ix = pos.x * params.unit - params.pad_x;
    int iy = pos.y * params.unit - params.pad_y;
    
    auto z_in = in + pos.z * params.input_width * params.input_height;
    auto S00 = get_input(z_in, ix + 0, iy + 0, params);
    auto S10 = get_input(z_in, ix + 1, iy + 0, params);
    auto S20 = get_input(z_in, ix + 2, iy + 0, params);
    auto S30 = get_input(z_in, ix + 3, iy + 0, params);
    auto S01 = get_input(z_in, ix + 0, iy + 1, params);
    auto S11 = get_input(z_in, ix + 1, iy + 1, params);
    auto S21 = get_input(z_in, ix + 2, iy + 1, params);
    auto S31 = get_input(z_in, ix + 3, iy + 1, params);
    auto S02 = get_input(z_in, ix + 0, iy + 2, params);
    auto S12 = get_input(z_in, ix + 1, iy + 2, params);
    auto S22 = get_input(z_in, ix + 2, iy + 2, params);
    auto S32 = get_input(z_in, ix + 3, iy + 2, params);
    auto S03 = get_input(z_in, ix + 0, iy + 3, params);
    auto S13 = get_input(z_in, ix + 1, iy + 3, params);
    auto S23 = get_input(z_in, ix + 2, iy + 3, params);
    auto S33 = get_input(z_in, ix + 3, iy + 3, params);
    
    auto m00 = +S00 - S02;
    auto m10 = +S10 - S12;
    auto m20 = +S20 - S22;
    auto m30 = +S30 - S32;
    auto m01 = +0.5 * S01 + 0.5 * S02;
    auto m11 = +0.5 * S11 + 0.5 * S12;
    auto m21 = +0.5 * S21 + 0.5 * S22;
    auto m31 = +0.5 * S31 + 0.5 * S32;
    auto m02 = -0.5 * S01 + 0.5 * S02;
    auto m12 = -0.5 * S11 + 0.5 * S12;
    auto m22 = -0.5 * S21 + 0.5 * S22;
    auto m32 = -0.5 * S31 + 0.5 * S32;
    auto m03 = -S01 + S03;
    auto m13 = -S11 + S13;
    auto m23 = -S21 + S23;
    auto m33 = -S31 + S33;
    
    int dst_x_origin = pos.z;
    int dst_y_origin = params.unit_width * pos.y + pos.x;
    int dst_y_stride = params.input_slice * 4;
    int dst_y        = dst_y_origin / 4;
    int dst_x        = dst_y_origin % 4 + 4 * dst_x_origin;
    int src_height   = UP_DIV(params.unit_width * params.unit_height, 4);
    int stride       = src_height * dst_y_stride;
    auto xy_out = out + dst_y * dst_y_stride + dst_x;
    *xy_out =  +m00 - m20;
    xy_out += stride; *xy_out =  +0.5 * m10 + 0.5 * m20;
    xy_out += stride; *xy_out =  -0.5 * m10 + 0.5 * m20;
    xy_out += stride; *xy_out =  -m10 + m30;
    xy_out += stride; *xy_out =  +m01 - m21;
    xy_out += stride; *xy_out =  +0.5 * m11 + 0.5 * m21;
    xy_out += stride; *xy_out =  -0.5 * m11 + 0.5 * m21;
    xy_out += stride; *xy_out =  -m11 + m31;
    xy_out += stride; *xy_out =  +m02 - m22;
    xy_out += stride; *xy_out=  +0.5 * m12 + 0.5 * m22;
    xy_out += stride; *xy_out =  -0.5 * m12 + 0.5 * m22;
    xy_out += stride; *xy_out =  -m12 + m32;
    xy_out += stride; *xy_out =  +m03 - m23;
    xy_out += stride; *xy_out =  +0.5 * m13 + 0.5 * m23;
    xy_out += stride; *xy_out =  -0.5 * m13 + 0.5 * m23;
    xy_out += stride; *xy_out =  -m13 + m33;
}

static inline void set_output(constant MetalWinogradParams &params, device ftype4 *output, int x, int y, ftype4 value) {
    output[y * params.output_width + x] = activate(value, params.activation);
}

kernel void winograd_transform_dest2_3_1(const device ftype4 *in            [[buffer(0)]],
                                         device ftype4 *out                 [[buffer(1)]],
                                         const device ftype4 *biasTerms     [[buffer(2)]],
                                         constant MetalWinogradParams &params   [[buffer(3)]],
                                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x >= params.unit_width || (int)gid.y >= params.unit_height)
        return;
    auto pos = int3(gid);
    
    int dst_w        = UP_DIV(params.unit_width * params.unit_height, 4);
    int dst_x_origin = params.unit_width * pos.y + pos.x;
    int dst_x        = dst_x_origin / 4;
    int dst_y        = 4 * pos.z + dst_x_origin % 4;
    int dst_y_stride = dst_w * 16;
    auto xy_in = in + dst_y * dst_y_stride + dst_x;
    
    auto S00 = *xy_in; xy_in += dst_w;
    auto S10 = *xy_in; xy_in += dst_w;
    auto S20 = *xy_in; xy_in += dst_w;
    auto S30 = *xy_in; xy_in += dst_w;
    auto S01 = *xy_in; xy_in += dst_w;
    auto S11 = *xy_in; xy_in += dst_w;
    auto S21 = *xy_in; xy_in += dst_w;
    auto S31 = *xy_in; xy_in += dst_w;
    auto S02 = *xy_in; xy_in += dst_w;
    auto S12 = *xy_in; xy_in += dst_w;
    auto S22 = *xy_in; xy_in += dst_w;
    auto S32 = *xy_in; xy_in += dst_w;
    auto S03 = *xy_in; xy_in += dst_w;
    auto S13 = *xy_in; xy_in += dst_w;
    auto S23 = *xy_in; xy_in += dst_w;
    auto S33 = *xy_in;
    
    auto m00 = +S00 + S01 + S02;
    auto m10 = +S10 + S11 + S12;
    auto m20 = +S20 + S21 + S22;
    auto m30 = +S30 + S31 + S32;
    auto m01 = +S01 - S02 + S03;
    auto m11 = +S11 - S12 + S13;
    auto m21 = +S21 - S22 + S23;
    auto m31 = +S31 - S32 + S33;
    
    // write output
    auto b4 = params.has_bias? biasTerms[int(pos.z)] : ftype4(Zero4);
    int oy = pos.y * params.unit;
    int ox = pos.x * params.unit;
    auto z_out = out + pos.z * params.output_width * params.output_height;
    
    /* if true */ {
        set_output(params, z_out, ox + 0, oy + 0, b4 + m00 + m10 + m20);
    }
    if (ox + 1 < params.output_width) {
        set_output(params, z_out, ox + 1, oy + 0, b4 + m10 - m20 + m30);
    }
    if (oy + 1 < params.output_height) {
        set_output(params, z_out, ox + 0, oy + 1, b4 + m01 + m11 + m21);
    }
    if (ox + 1 < params.output_width && oy + 1 < params.output_height) {
        set_output(params, z_out, ox + 1, oy + 1, b4 + m11 - m21 + m31);
    }
}
