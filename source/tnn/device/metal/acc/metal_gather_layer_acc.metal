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

#include <metal_math>
#include <metal_stdlib>
#include "tnn/device/metal/acc/metal_common.metal"

using namespace metal;
kernel void gather_axis_1(const device ftype *src                   [[buffer(0)]],
                          const device int *gather_indices          [[buffer(1)]],
                                device ftype *dst                   [[buffer(2)]],
                                constant MetalGatherParams &params  [[buffer(3)]],
                                uint3 gid                           [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.inner_size, params.output_axis_size, params.outer_size)))
        return;

    int output_slice = (int)gid.y / 4;
    int output_c     = (int)gid.y % 4;
    int index_out = (((int)gid.z*params.output_slice + output_slice)*params.inner_size + (int)gid.x)*4 + output_c;

    int input_axis_index = gather_indices[(int)gid.y];
    int input_slice = input_axis_index / 4;
    int input_c     = input_axis_index % 4;
    int index_in = (((int)gid.z*params.input_slice + input_slice)*params.inner_size + (int)gid.x)*4 + input_c;

    dst[index_out] = src[index_in];
}

kernel void gather_common(const device ftype4 *src                  [[buffer(0)]],
                          const device int *gather_indices          [[buffer(1)]],
                                device ftype4 *dst                  [[buffer(2)]],
                                constant MetalGatherParams &params  [[buffer(3)]],
                                uint3 gid                           [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.inner_size, params.output_axis_size, params.outer_size)))
        return;
    
    int index_out = ((int)gid.z*params.output_axis_size + (int)gid.y)*params.inner_size + (int)gid.x;

    int input_axis_index = gather_indices[gid.y];
    int index_in = ((int)gid.z*params.input_axis_size + input_axis_index)*params.inner_size + (int)gid.x;

    dst[index_out] = src[index_in];
}

kernel void gather_common_nchw(const device ftype *src                  [[buffer(0)]],
                          const device int *gather_indices          [[buffer(1)]],
                                device ftype *dst                  [[buffer(2)]],
                                constant MetalGatherParams &params  [[buffer(3)]],
                                uint3 gid                           [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.inner_size, params.output_axis_size, params.outer_size)))
        return;
    
    int index_out = ((int)gid.z*params.output_axis_size + (int)gid.y)*params.inner_size + (int)gid.x;

    int input_axis_index = gather_indices[gid.y];
    int index_in = ((int)gid.z*params.input_axis_size + input_axis_index)*params.inner_size + (int)gid.x;

    dst[index_out] = src[index_in];
}