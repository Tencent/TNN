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
#define COMPARE_SET_FLAG(f, v1, v2, m)          \
    do {                                        \
        if (m == 0) {                           \
            f = bool4(v1 <= v2);                \
        } else {                                \
            f = bool4(v1 >= v2);                \
        }                                       \
    } while(0)

#define REDUCE_VEC4(vec, op)                    \
        op(vec.x, op(vec.y, op(vec.z, vec.w)))

kernel void argmax_or_min_common(const device ftype4 *src                   [[buffer(0)]],
                                 device       int4   *dst                   [[buffer(1)]],
                                 constant MetalArgMaxOrMinParams &params    [[buffer(2)]],
                                 uint3 gid                                  [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.inner_size, params.outer_size, 1)))
        return;

    int index_out = (int)gid.y * params.inner_size + (int)gid.x;
    int index_in  = (int)gid.y * params.inner_size * params.reduce_size + (int)gid.x;

    ftype4 guard_value = src[index_in];
    int4   guard_index = int4(0);
    auto   flag        = bool4(false);
    for(int r=1; r<params.reduce_size; ++r) {
        index_in += params.inner_size;
        ftype4 val = src[index_in];
        int4   idx = int4(r);
        COMPARE_SET_FLAG(flag, guard_value, val, params.mode);
        guard_value = select(val, guard_value, flag);
        guard_index = select(idx, guard_index, flag);
    }

    dst[index_out] = guard_index;
}

kernel void argmax_or_min_channel(const device ftype4 *src                  [[buffer(0)]],
                                  device       int4   *dst                  [[buffer(1)]],
                                  constant MetalArgMaxOrMinParams &params   [[buffer(2)]],
                                  uint3 gid                                 [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.inner_size, params.outer_size, 1)))
        return;

    ftype pad_val = params.mode == 0 ? ftype(FLT_MAX):ftype(-FLT_MAX);

    int index_out = (int)gid.y * params.inner_size + (int)gid.x;
    int index_in  = (int)gid.y * params.inner_size * params.reduce_size + (int)gid.x;

    ftype4 guard_value = ftype4(pad_val);
    int4   guard_index = int4(-1);

    auto reduce_c4 = params.input_channel / 4;
    auto reduce_r4 = params.input_channel % 4;
    int4 idx = int4(0, 1, 2, 3);
    auto flag = bool4(false);
    for(int rc=0; rc<reduce_c4; ++rc) {
        ftype4 val = src[index_in];
        COMPARE_SET_FLAG(flag, guard_value, val, params.mode);
        guard_value = select(val, guard_value, flag);
        guard_index = select(idx, guard_index, flag);

        index_in += params.inner_size;
        idx += 4;
    }
    if (reduce_r4 != 0) {
        ftype4 r4_value = src[index_in];
        switch(reduce_r4) {
            case 1:
                r4_value = ftype4(r4_value.x, pad_val, pad_val, pad_val);
                break;
            case 2:
                r4_value = ftype4(r4_value.x, r4_value.y, pad_val, pad_val);
                break;
            case 3:
                r4_value = ftype4(r4_value.x, r4_value.y, r4_value.z, pad_val);
                break;
        }
        COMPARE_SET_FLAG(flag, guard_value, r4_value, params.mode);
        guard_value = select(r4_value, guard_value, flag);
        guard_index = select(idx, guard_index, flag);
    }
    // find the target value in ftype4
    ftype target_val = params.mode==0? REDUCE_VEC4(guard_value, min):REDUCE_VEC4(guard_value, max);
    auto eq  = (guard_value == target_val);
    idx = select(int4(params.input_channel), guard_index, eq);
    auto target_idx = REDUCE_VEC4(idx, min);

    dst[index_out] = int4(target_idx, 0, 0, 0);
}