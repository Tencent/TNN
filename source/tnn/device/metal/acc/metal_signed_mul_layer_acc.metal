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

#define ARITHMETIC

using namespace metal;
kernel void signed_mul_fused(const device ftype4 *in_ptr                           [[buffer(0)]],
                        device ftype4 *out_ptr                                   [[buffer(1)]],
                        constant MetalSignedMulParams& params       [[buffer(2)]],
                        uint3 gid                                                  [[thread_position_in_grid]]) {
    
    if (any(uint3(gid) >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    auto in  = in_ptr[(int)gid.z * params.input_slice * params.input_size  + (int)gid.y * params.input_size + (int)gid.x];
    auto z_out = out_ptr + (int)gid.z * params.output_slice * params.output_size + (int)gid.y * params.output_size + (int)gid.x;
    auto mul = in_ptr[(int)gid.z * params.input_slice * params.input_size  +  (int)gid.x];
    mul = mul.xxxx;
    
    auto temp = in - params.alpha;
    auto mul_temp  = mul - params.alpha;
    temp = sign(temp)*1;
    mul_temp = sign(mul_temp)*1;

    temp += params.beta;
    mul_temp += params.beta;
    temp *= params.gamma_inv;
    mul_temp *= params.gamma_inv;
    temp *= mul_temp;
    *z_out = temp;
}

/*
* Specialization for channel=4
*/
kernel void signed_mul_fused_channel4(const device ftype4 *in_ptr                           [[buffer(0)]],
                        device ftype4 *out_ptr                                   [[buffer(1)]],
                        constant MetalSignedMulParams& params       [[buffer(2)]],
                        uint3 gid                                                  [[thread_position_in_grid]]) {
    
    if (any(uint3(gid) >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    auto in  = in_ptr[(int)gid.z * params.input_slice * params.input_size  + (int)gid.x];
    auto z_out = out_ptr + (int)gid.z * params.output_slice * params.output_size + (int)gid.x;
    
    auto temp = in - params.alpha;
    temp = sign(temp)*1;

    temp += params.beta;
    temp *= params.gamma_inv;
    auto mul = temp.xxxx;
    temp *= mul;
    *z_out = temp;
}