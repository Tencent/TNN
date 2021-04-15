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
kernel void stride_slice_common(const device ftype4 *src                                    [[buffer(0)]],
                                                     device ftype4 *dst                                              [[buffer(1)]],
                                                     constant MetalStrideSliceParams& params      [[buffer(2)]],
                                                     uint3 gid                                                             [[thread_position_in_grid]]) {
    if (any(uint3(gid) >= uint3(params.output_width, params.output_height, params.batch*params.output_slice)))
        return;
    const int batch_out = gid.z / params.output_slice;
    const int slice_out = gid.z % params.output_slice;
    
    int index_out = (int)gid.z*params.output_size + (int)gid.y*params.output_width + (int)gid.x;
    
    const int4 channel_out = slice_out*4 + int4(0, 1, 2, 3);
    
    int batch_in = batch_out * params.stride_n + params.begin_n;
    int4 channel_in = channel_out * params.stride_c + params.begin_c;
    int4 slice_in = channel_in / 4;
    slice_in = min(slice_in, params.input_slice-1);
    int4 input_i = channel_in % 4;
    int height_in = (int)gid.y * params.stride_h + params.begin_h;
    int width_in = (int)gid.x * params.stride_w + params.begin_w;
    
    int4 index_in = batch_in * params.input_slice * params.input_size + slice_in * params.input_size + height_in * params.input_width + width_in;
    
    if (all(index_in ==  index_in.yzwx) && all(input_i ==  int4(0, 1, 2, 3))) {
        dst[index_out] = src[index_in[0]];
    } else {
        dst[index_out] = ftype4(
            src[index_in[0]][input_i[0]],
            src[index_in[1]][input_i[1]],
            src[index_in[2]][input_i[2]],
            src[index_in[3]][input_i[3]]
        );
    }
    
}

static uint3 linear_to_3d(uint idx, constant int shape[3]) {
    uint3 idx3d;
    idx3d.x = idx % shape[0];
    idx3d.y = (idx / shape[0]) % shape[1];
    idx3d.z = (idx / shape[0]) / shape[1];

    return idx3d;
}

template <int S, typename T>
static uint dot(T a, constant uint b[S]) {
    uint rst = 0;
    for(int i=0; i<S; ++i) rst += a[i] * b[i];
    return rst;
}

template <int S, typename T>
static uint dot(T a, T b) {
    uint rst = 0;
    for(int i=0; i<S; ++i) rst += a[i] * b[i];
    return rst;
}

template <int S, typename T>
static T mul(T a, constant int b[S]) {
    T rst;
    for(int i=0; i<S; ++i) rst[i] = a[i] * b[i];
    return rst;
}

template <int S, typename T>
static T add(T a, constant int b[S]) {
    T rst;
    for(int i=0; i<S; ++i) rst[i] = a[i] + b[i];
    return rst;
}


kernel void stride_slice_common_dim6(const device ftype4 *src                                    [[buffer(0)]],
                                                     device ftype4 *dst                                              [[buffer(1)]],
                                                     constant MetalStrideSliceParamsV2& params      [[buffer(2)]],
                                                     uint3 gid                                                             [[thread_position_in_grid]]) {
    if (any(uint3(gid) >= uint3(params.output_width, params.output_height, params.batch*params.output_slice)))
        return;

    const int batch_out = gid.z / params.output_slice;
    const int slice_out = gid.z % params.output_slice;
    uint3 idx_high = uint3(gid.y, slice_out, batch_out);
    uint3 idx_low  = linear_to_3d(gid.x, params.shape3d_low);
    
    int index_out = dot<3>(idx_high, uint3(params.output_width, params.output_size, params.output_size*params.output_slice));
    index_out    += dot<3>(idx_low,  uint3(1, params.shape3d_low[0], params.shape3d_low[0]*params.shape3d_low[1]));

    const int4 channel_out = slice_out*4 + int4(0, 1, 2, 3);
    int4 channel_in = channel_out * params.strides_high[1] + params.begins_high[1];
    int4 slice_in = channel_in / 4;
    int4 input_i  = channel_in % 4;
    slice_in = min(slice_in, params.input_slice-1);

    uint3 input_idx_low = add<3>(mul<3>(idx_low, params.strides_low), params.begins_low);
    uint3 input_idx_high;
    input_idx_high.x = idx_high[0] * params.strides_high[0] + params.begins_high[0];
    input_idx_high.z = idx_high[2] * params.strides_high[2] + params.begins_high[2];
    if (all(slice_in == slice_in.yzwx) && all(input_i == int4(0, 1, 2, 3))) {
        input_idx_high.y = slice_in[0];
        int index_in = dot<3>(input_idx_low, uint3(1, params.input_shape3d_low[0],  params.input_shape3d_low[0] * params.input_shape3d_low[1]));
        index_in += dot<3>(input_idx_high, uint3(params.input_width, params.input_size, params.input_size*params.input_slice));

        dst[index_out] = src[index_in];
    } else {
        int4 index_in = dot<3>(input_idx_low, uint3(1, params.input_shape3d_low[0],  params.input_shape3d_low[0] * params.input_shape3d_low[1]));
        for(int i=0; i<4; ++i) {
            input_idx_high.y = slice_in[i];
            index_in[i] += dot<3>(input_idx_high, uint3(params.input_width, params.input_size, params.input_size*params.input_slice));;
        }
        dst[index_out] = ftype4(
            src[index_in[0]][input_i[0]],
            src[index_in[1]][input_i[1]],
            src[index_in[2]][input_i[2]],
            src[index_in[3]][input_i[3]]
        );
    }
    
}