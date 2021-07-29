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
template <typename Src, typename Dst>
static void cast_from_to(const device Src *src,
                               device Dst *dst,
                               uint index) {
    dst[index] = static_cast<Dst>(src[index]);
}

kernel void cast_same_bytes2(const device half4 *src                   [[buffer(0)]],
                                device half4 *dst                   [[buffer(1)]],
                                constant MetalCastParams &params  [[buffer(2)]],
                                uint3 gid                           [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.input_size, params.input_slice, params.batch)))
        return;

    uint index = (gid.z * params.input_slice + gid.y) * params.input_size + gid.x;
    cast_from_to<half4, half4>(src, dst, index);
}

kernel void cast_same_bytes4(const device float4 *src                   [[buffer(0)]],
                                device float4 *dst                   [[buffer(1)]],
                                constant MetalCastParams &params  [[buffer(2)]],
                                uint3 gid                           [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.input_size, params.input_slice, params.batch)))
        return;

    uint index = (gid.z * params.input_slice + gid.y) * params.input_size + gid.x;
    cast_from_to<float4, float4>(src, dst, index);
}

kernel void cast_ftype_to_int32(const device ftype4 *src            [[buffer(0)]],
                                device int4 *dst                   [[buffer(1)]],
                                constant MetalCastParams &params  [[buffer(2)]],
                                uint3 gid                           [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.input_size, params.input_slice, params.batch)))
        return;

    uint index = (gid.z * params.input_slice + gid.y) * params.input_size + gid.x;
    cast_from_to<ftype4, int4>(src, dst, index);

}

kernel void cast_int32_to_ftype(const device int4 *src            [[buffer(0)]],
                                device ftype4 *dst                   [[buffer(1)]],
                                constant MetalCastParams &params  [[buffer(2)]],
                                uint3 gid                           [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.input_size, params.input_slice, params.batch)))
        return;

    uint index = (gid.z * params.input_slice + gid.y) * params.input_size + gid.x;
    cast_from_to<int4, ftype4>(src, dst, index);

}

kernel void cast_int32_to_uint32(const device int4 *src            [[buffer(0)]],
                                device uint4 *dst                   [[buffer(1)]],
                                constant MetalCastParams &params  [[buffer(2)]],
                                uint3 gid                           [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.input_size, params.input_slice, params.batch)))
        return;

    uint index = (gid.z * params.input_slice + gid.y) * params.input_size + gid.x;
    cast_from_to<int4, uint4>(src, dst, index);

}

kernel void cast_uint32_to_int32(const device uint4 *src            [[buffer(0)]],
                                device int4 *dst                   [[buffer(1)]],
                                constant MetalCastParams &params  [[buffer(2)]],
                                uint3 gid                           [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.input_size, params.input_slice, params.batch)))
        return;

    uint index = (gid.z * params.input_slice + gid.y) * params.input_size + gid.x;
    cast_from_to<uint4, int4>(src, dst, index);

}