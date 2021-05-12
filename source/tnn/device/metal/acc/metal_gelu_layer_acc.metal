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

static float4 fast_erf_approximation(float4 x) {
    //use x*x instead of pow(x, 2), see  https://www.zhihu.com/question/60172486
    auto t = 1 / (1 + 0.5 * fabs(x));
    
    auto t_2 = t * t;
    auto t_3 = t_2 * t;
    auto t_4 = t_3 * t;
    auto t_5 = t_4 * t;
    auto t_6 = t_5 * t;
    auto t_7 = t_6 * t;
    auto t_8 = t_7 * t;
    auto t_9 = t_8 * t;
    
    auto v = t * exp(-x * x - 1.26551223 + 1.00002368 * t + 0.37409196 * t_2 + 0.09678418 * t_3 -
                     0.18628806 * t_4 + 0.27886807 * t_5 - 1.13520398 * t_6 +
                     1.48851587 * t_7 - 0.82215223 * t_8 + 0.17087277 * t_9);
    
    bool4 gt_zero = x >= float4(Zero4);
    return select(v-1, 1-v, gt_zero);
}

kernel void gelu(const device ftype4 *in                 [[buffer(0)]],
                         device ftype4 *out                         [[buffer(1)]],
                        constant MetalParams& params    [[buffer(2)]],
                         uint3 gid                                        [[thread_position_in_grid]]) {
    if (any(uint3(gid) >= uint3(params.output_size, params.output_slice, params.batch)))
        return;
    
    auto z_in  = in  + (int)gid.z * params.input_slice * params.input_size  + (int)gid.y * params.input_size + (int)gid.x;
    auto z_out = out + (int)gid.z *  params.output_slice*  params.output_size + (int)gid.y * params.output_size + (int)gid.x;

    auto x = float4(*z_in);
    *z_out = ftype4(0.5f * x * (fast_erf_approximation(x*0.707106793288165f) + 1.0f));
}
