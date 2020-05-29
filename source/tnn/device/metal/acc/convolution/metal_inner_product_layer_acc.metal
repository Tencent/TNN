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

kernel void inner_product(const device ftype4 *in                                      [[buffer(0)]],
                                         device ftype4 *out                                              [[buffer(1)]],
                                         constant MetalInnerProductParams & params  [[buffer(2)]],
                                         const device ftype4x4 *wt                                  [[buffer(3)]],
                                         const device ftype4 *biasTerms                         [[buffer(4)]],
                                         uint3 gid                                                             [[thread_position_in_grid]]) {
    if ((int)gid.x >= params.output_size || (int)gid.y >= params.output_slice || (int)gid.z >= params.batch) return;
    
    auto xy_wt  = wt                                                    + (int)gid.y * params.input_slice * params.input_size;
    auto xy_in  = in  + (int)gid.z * params.input_slice  * params.input_size  + (int)gid.x;
    auto xy_out = out + (int)gid.z * params.output_slice * params.output_size + (int)gid.y * params.output_size + (int)gid.x;
    
    auto result = params.has_bias ? float4(biasTerms[gid.y]) : float4(Zero4);
    for (auto z = 0; z < params.input_slice*params.input_size; z++) {
            result += float4(xy_in[z]) * float4x4(xy_wt[z]);
    }
    *xy_out = activate(ftype4(result), params.activation);
}
