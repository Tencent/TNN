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

kernel void nc4hw4_buffer_nchw_buffer(const device ftype4 *src                  [[buffer(0)]],
                                      device ftype *dst                        [[buffer(1)]],
                                      constant MetalImageConverterParams &params [[buffer(2)]],
                                      uint3 gid                                [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.size, params.slice, params.batch)))
        return;

    const int index_in =  (int)gid.z*params.slice*params.size + (int)gid.y*params.size + (int)gid.x;

    int channel_out = gid.y * 4;
    int index_out = ((int)gid.z*params.channel + channel_out)*params.size + (int)gid.x;

    ftype4 data  = src[index_in];

    dst[index_out] = data.x;
    if (channel_out + 3 < params.channel) {
        dst[index_out+params.size]   = data.y;
        dst[index_out+params.size*2] = data.z;
        dst[index_out+params.size*3] = data.w;
    } else if (channel_out + 2 < params.channel) {
        dst[index_out+params.size]   = data.y;
        dst[index_out+params.size*2] = data.z;
    } else if (channel_out + 1 < params.channel) {
        dst[index_out+params.size]   = data.y;
    }
}

kernel void nchw_buffer_nc4hw4_buffer(const device ftype *src                   [[buffer(0)]],
                                      device ftype4 *dst                       [[buffer(1)]],
                                      constant MetalImageConverterParams& params [[buffer(2)]],
                                      uint3 gid                                [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.size, params.slice, params.batch)))
        return;

    int channel_in= gid.y * 4;
    int index_in = ((int)gid.z*params.channel + channel_in)*params.size + (int)gid.x;

    const int index_out =  (int)gid.z*params.slice*params.size + (int)gid.y * params.size + (int)gid.x;

    ftype4 data  = ftype4(Zero4);

    data.x = src[index_in];
    if (channel_in + 3 < params.channel) {
        data.y = src[index_in + params.size];
        data.z = src[index_in + params.size*2];
        data.w = src[index_in + params.size*3];
    } else if (channel_in + 2 < params.channel) {
        data.y = src[index_in + params.size];
        data.z = src[index_in + params.size*2];
    } else if (channel_in + 1 < params.channel) {
        data.y = src[index_in + params.size];
    }

    dst[index_out] = data;
}

kernel void nc4hw4_buffer_nchw_buffer_int32(const device int4 *src                  [[buffer(0)]],
                                      device int *dst                        [[buffer(1)]],
                                      constant MetalImageConverterParams &params [[buffer(2)]],
                                      uint3 gid                                [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.size, params.slice, params.batch)))
        return;

    const int index_in =  (int)gid.z*params.slice*params.size + (int)gid.y*params.size + (int)gid.x;

    int channel_out = gid.y * 4;
    int index_out = ((int)gid.z*params.channel + channel_out)*params.size + (int)gid.x;

    auto data  = src[index_in];

    dst[index_out] = data.x;
    if (channel_out + 3 < params.channel) {
        dst[index_out+params.size]   = data.y;
        dst[index_out+params.size*2] = data.z;
        dst[index_out+params.size*3] = data.w;
    } else if (channel_out + 2 < params.channel) {
        dst[index_out+params.size]   = data.y;
        dst[index_out+params.size*2] = data.z;
    } else if (channel_out + 1 < params.channel) {
        dst[index_out+params.size]   = data.y;
    }
}

kernel void nchw_buffer_nc4hw4_buffer_int32(const device int *src                   [[buffer(0)]],
                                      device int4 *dst                       [[buffer(1)]],
                                      constant MetalImageConverterParams& params [[buffer(2)]],
                                      uint3 gid                                [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.size, params.slice, params.batch)))
        return;

    int channel_in= gid.y * 4;
    int index_in = ((int)gid.z*params.channel + channel_in)*params.size + (int)gid.x;

    const int index_out =  (int)gid.z*params.slice*params.size + (int)gid.y * params.size + (int)gid.x;

    auto data  = int4(Zero4);

    data.x = src[index_in];
    if (channel_in + 3 < params.channel) {
        data.y = src[index_in + params.size];
        data.z = src[index_in + params.size*2];
        data.w = src[index_in + params.size*3];
    } else if (channel_in + 2 < params.channel) {
        data.y = src[index_in + params.size];
        data.z = src[index_in + params.size*2];
    } else if (channel_in + 1 < params.channel) {
        data.y = src[index_in + params.size];
    }

    dst[index_out] = data;
}
