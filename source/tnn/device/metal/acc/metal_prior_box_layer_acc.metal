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
kernel void copy_buffer_2_buffer(const device ftype4 *src                    [[buffer(0)]],
                                                    device ftype4 *dst                             [[buffer(1)]],
                                                    constant MetalParams& params      [[buffer(2)]],
                                                    uint3 gid                                            [[thread_position_in_grid]]) {
    if (any(uint3(gid) >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    auto index = (int)gid.z * params.output_slice * params.output_size + (int)gid.y * params.output_size + (int)gid.x;
    dst[index] = src[index];
}


