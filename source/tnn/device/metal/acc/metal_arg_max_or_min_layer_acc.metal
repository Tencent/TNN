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
kernel void argmax_or_min_common(const device ftype4 *src                   [[buffer(0)]],
                                 device       ftype4 *dst                   [[buffer(1)]],
                                 constant MetalArgMaxOrMinParams &params    [[buffer(2)]],
                                 uint3 gid                                  [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.inner_size, params.outer_size, 1)))
        return;

    int index_out = (int)gid.y * params.inner_size + (int)gid.x;
    int index_in  = (int)gid.y * params.inner_size * params.reduce_size + (int)gid.x;

    ftype4 guard_value = src[index_in];
    int4   guard_index = int4(0);
    for(int r=1; r<params.reduce_size; ++r) {
        index_in += params.inner_size;
        ftype4 val = src[index_in];
        int4   idx = int4(r);
        auto flag = bool4(false);
        if (params.mode == 0) {
            //argmin
            flag = bool4(guard_value <= val);
        } else {
            //argmax
            flag = bool4(guard_value >= val);
        }
        guard_value = select(val, guard_value, flag);
        guard_index = select(idx, guard_index, flag);
    }

    dst[index_out] = ftype4(guard_index);

}

kernel void argmax_or_min_channel(const device ftype4 *src                  [[buffer(0)]],
                                  device       ftype4 *dst                  [[buffer(1)]],
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
    for(int rc=0; rc<reduce_c4; ++rc) {
        ftype4 val = src[index_in];
        auto flag = bool4(false);
        if (params.mode == 0) {
            //argmin
            flag = bool4(guard_value <= val);
        } else {
            //argmax
            flag = bool4(guard_value >= val);
        }
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
        auto flag = bool4(false);
        if (params.mode == 0) {
            //argmin
            flag = bool4(guard_value <= r4_value);
        } else {
            //argmax
            flag = bool4(guard_value >= r4_value);
        }
        guard_value = select(r4_value, guard_value, flag);
        guard_index = select(idx, guard_index, flag);
    }
    // find the target value in ftype4
    ftype target_idx = params.input_channel;
    {
        if (params.mode == 0) {
            ftype min_val = fmin(guard_value.x, fmin(guard_value.y, fmin(guard_value.z, guard_value.w)));
            if (min_val == guard_value.x)
                target_idx = fmin(target_idx, guard_index.x);
            if (min_val == guard_value.y)
                target_idx = fmin(target_idx, guard_index.y);
            if (min_val == guard_value.z)
                target_idx = fmin(target_idx, guard_index.z);
            if (min_val == guard_value.w)
                target_idx = fmin(target_idx, guard_index.w);
        } else {
            ftype max_val = fmax(guard_value.x, fmax(guard_value.y, fmax(guard_value.z, guard_value.w)));
            if (max_val == guard_value.x)
                target_idx = fmin(target_idx, guard_index.x);
            if (max_val == guard_value.y)
                target_idx = fmin(target_idx, guard_index.y);
            if (max_val == guard_value.z)
                target_idx = fmin(target_idx, guard_index.z);
            if (max_val == guard_value.w)
                target_idx = fmin(target_idx, guard_index.w);
        }
    }
    dst[index_out] = ftype4(target_idx, 0, 0, 0);
}