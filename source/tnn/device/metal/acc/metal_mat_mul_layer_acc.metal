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

kernel void matmul_common(const device ftype *mat_a                                      [[buffer(0)]],
                           const device ftype *mat_b                                              [[buffer(1)]],
                                        device ftype *mat_c                                [[buffer(2)]],
                                        constant MetalMatMulParams & params  [[buffer(3)]],
                                        uint3 gid                                                             [[thread_position_in_grid]]) {
    if ((int)gid.x >= params.K || (int)gid.y >= params.M || (int)gid.z >= params.batch_c) return;
        int batch_a = (int)gid.z % params.batch_a;
        int batch_b = (int)gid.z % params.batch_b;

        auto ptr_a = mat_a + (batch_a * params.M + (int)gid.y) * params.N;
        auto ptr_b = mat_b + (batch_b * params.K * params.N) + (int)gid.x;

        ftype result = 0;
        for(int i = 0; i < params.N; i++){
            auto a_t =  ptr_a + i;
            auto b_t =  ptr_b + i * params.K;
            result += ftype(*a_t) * ftype(*b_t);
        }

        mat_c[((int)gid.z*params.M + (int)gid.y) * params.K + (int)gid.x] = ftype(result);
}
