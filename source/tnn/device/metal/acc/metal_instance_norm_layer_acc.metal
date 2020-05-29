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
kernel void instance_norm(const device ftype4 *src                       [[buffer(0)]],
                                            device ftype4 *dst                                [[buffer(1)]],
                                            constant MetalParams& params         [[buffer(2)]],
                                            const device ftype4 *scales                 [[buffer(3)]],
                                            const device ftype4 *biases                [[buffer(4)]],
                                            uint3 gid                                              [[thread_position_in_grid]],
                                            uint t_index                                         [[thread_index_in_threadgroup]]) {
    if (any(gid >= uint3(params.input_size, 1, params.batch*params.input_slice)))
        return;
    
    auto index_c = (int)gid.z * params.input_size;
    auto index_slice = (int)gid.z % params.input_slice;
    
    const int max_index = min(32, params.input_size);
    
    //do not use setThreadgroupMemoryLength, unknown bug will raise
    threadgroup float4 x_group[32];
    threadgroup float4 x2_group[32];
    
    //compute sum x x2
    float4 sum_x = float4(0);
    float4 sum_x2 = float4(0);
    for (int index = t_index; index < params.input_size; index+=32) {
        auto temp = float4(src[index + index_c]);
        sum_x += temp;
        sum_x2 += temp*temp;
    }
    x_group[t_index] = sum_x;
    x2_group[t_index] = sum_x2;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    //compute mean x x2
    if (t_index == 0) {
        sum_x = float4(0);
        sum_x2 = float4(0);
        for (int index = 0; index < max_index; index++) {
            sum_x += x_group[index];
            sum_x2 += x2_group[index];
        }
        auto mean_x = sum_x / params.input_size;
        auto mean_x2 = sum_x2 / params.input_size;

        auto variance = mean_x2 - mean_x*mean_x;
        variance = max(variance, 0);
        variance =1.0f / sqrt(variance + 0.00001f);

        auto k = float4(scales[index_slice]);
        variance *= (k);
        auto b = float4(biases[index_slice]);
        b -= mean_x * variance;
        
        x_group[0] = variance;
        x2_group[0] = b;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    auto scale_final = x_group[0];
    auto bias_final = x2_group[0];
    for (int index = t_index; index < params.input_size; index+=32) {
        dst[index + index_c] = ftype4(float4(src[index + index_c]) * scale_final + bias_final);
    }
}
