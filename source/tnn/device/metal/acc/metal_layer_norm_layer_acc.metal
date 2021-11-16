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
kernel void layer_norm(const device ftype *src                       [[buffer(0)]],
                                           const device ftype *scales                 [[buffer(1)]],
                                           const device ftype *biases                [[buffer(2)]],
                                            device ftype *dst                                [[buffer(3)]],
                                            constant MetalLayerNormParams& params         [[buffer(4)]],
 
                                            uint3 gid                                              [[thread_position_in_grid]],
                                            uint t_index                                         [[thread_index_in_threadgroup]]) {
    // channel_area, 1, channels
    if (any(gid >= uint3(params.channel_area, 1, params.channels)))
        return;
    
    auto index_c = (int)gid.z * params.channel_area;
    const int max_index = min(32, params.channel_area);
    
    //do not use setThreadgroupMemoryLength, unknown bug will raise
    threadgroup float x_group[32];
    threadgroup float x2_group[32];
    
    //compute sum x x2
    float sum_x  = 0.f;
    float sum_x2 = 0.f;
    for (int index = t_index; index < params.channel_area; index+=32) {
        auto temp = float(src[index + index_c]);
        sum_x  += temp;
        sum_x2 += temp*temp;
    }
    x_group[t_index]  = sum_x;
    x2_group[t_index] = sum_x2;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    //compute mean x x2
    if (t_index == 0) {
        sum_x  = 0.f;
        sum_x2 = 0.f;
        for (int index = 0; index < max_index; index++) {
            sum_x  += x_group[index];
            sum_x2 += x2_group[index];
        }
        auto mean_x  = sum_x  / params.channel_area;
        auto mean_x2 = sum_x2 / params.channel_area;

        auto variance = mean_x2 - mean_x*mean_x;
        variance = max(variance, 0.f);
        variance = 1.0f / sqrt(variance + params.eps);

        x_group[0]  = mean_x;
        x2_group[0] = variance;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int index = t_index; index < params.channel_area; index+=32) {
        float scale = float(scales[index]);
        float bias  = float(biases[index]);
        bias -= x_group[0] * x2_group[0] * scale;

        dst[index + index_c] = src[index + index_c] * x2_group[0] * scale + bias;
    }
}
