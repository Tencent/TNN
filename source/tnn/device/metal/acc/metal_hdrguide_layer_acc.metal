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
kernel void hdr_guide(const device ftype4 *src             [[buffer(0)]],
                       device ftype4 *dst                   [[buffer(1)]],
                       constant MetalParams& params         [[buffer(2)]],
                       const device ftype4x4& ccm_weights          [[buffer(3)]],
                       const device ftype4& ccm_biases          [[buffer(4)]],
                      const device ftype4 *slopes          [[buffer(5)]],
                      const device ftype4 *shifts          [[buffer(6)]],
                      const device ftype4& projection          [[buffer(7)]],
                       uint2 gid                            [[thread_position_in_grid]]) {
    if (any(gid >= uint2(params.output_size, params.output_slice*params.batch)))
        return;
    
    auto index = (int)gid.y * params.input_size  + (int)gid.x;
    auto index_out = (int)gid.y * params.output_size  + (int)gid.x;
    
    // use ccm, create new r, g, b value
    auto src_data = src[index];
    ftype4 result = src_data * ccm_weights + ccm_biases;
    
    // use slope and shifts per channel
    ftype4 guide_result  = slopes[0] * max(result - shifts[0], Zero4);
    guide_result += slopes[1] * max(result - shifts[1], Zero4);
    guide_result += slopes[2] * max(result - shifts[2], Zero4);
    guide_result += slopes[3] * max(result - shifts[3], Zero4);

    // channel mix
    ftype4 guide_value = ftype4(dot(projection.xyz, guide_result.xyz) + projection.w, 0, 0, 0);
    guide_value = clamp(guide_value, 0, 1);
    dst[index_out] = guide_value;
}

