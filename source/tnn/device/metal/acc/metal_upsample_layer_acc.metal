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

kernel void upsample_nearest(const device ftype4 *src                [[buffer(0)]],
                             device ftype4 *dst                      [[buffer(1)]],
                             constant MetalUpsampleParams &params    [[buffer(2)]],
                             uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch)))
        return;
    
    int src_x = floor(gid.x * params.scale_x);
    int src_y = floor(gid.y * params.scale_y);
    
    int index_dst = int(gid.z) * params.output_size + int(gid.y) * params.output_width + int(gid.x);
    int index_src = int(gid.z) * params.input_size  + src_y * params.input_width + src_x;
    
    dst[index_dst] = src[index_src];
}

kernel void upsample_bilinear_align(const device ftype4 *src                [[buffer(0)]],
                                    device ftype4 *dst                      [[buffer(1)]],
                                    constant MetalUpsampleParams &params    [[buffer(2)]],
                                    uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch)))
        return;
    
    float srcX = gid.x * params.scale_x, srcY = gid.y * params.scale_y;
    int left = floor(srcX), right = min(left + 1, params.input_width - 1);
    int top = floor(srcY), bottom = min(top + 1, params.input_height - 1);
    
    float x2_factor = srcX - left;
    float y2_factor = srcY - top;
    float x1_factor = 1 - x2_factor;
    float y1_factor = 1 - y2_factor;
    
    auto in_z        = src + gid.z * params.input_size;
    auto in_top      = in_z + top * params.input_width;
    auto in_bottom   = in_z + bottom * params.input_width;
    auto tl = ftype4(in_top[left])     * x1_factor * y1_factor;
    auto tr = ftype4(in_top[right])    * x2_factor * y1_factor;
    auto bl = ftype4(in_bottom[left])  * x1_factor * y2_factor;
    auto br = ftype4(in_bottom[right]) * x2_factor * y2_factor;
    dst[int(gid.z) * params.output_size + int(gid.y) * params.output_width + int(gid.x)] = ftype4(tl + tr + bl + br);
}


kernel void upsample_bilinear_noalign(const device ftype4 *in                 [[buffer(0)]],
                                      device ftype4 *out                      [[buffer(1)]],
                                      constant MetalUpsampleParams &params    [[buffer(2)]],
                                      uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch)))
        return;
    
    float srcX = max(params.scale_x*(gid.x+0.5f) - 0.5f, 0.0f);
    float srcY = max(params.scale_y*(gid.y+0.5f) - 0.5f, 0.0f);
    int left = floor(srcX), right = min(left + 1, params.input_width - 1);
    int top = floor(srcY), bottom = min(top + 1, params.input_height - 1);
    
    float x2_factor = srcX - left;
    float y2_factor = srcY - top;
    float x1_factor = 1 - x2_factor;
    float y1_factor = 1 - y2_factor;
    
    auto in_z        = in + gid.z * params.input_size;
    auto in_top      = in_z + top * params.input_width;
    auto in_bottom   = in_z + bottom * params.input_width;
    auto tl = ftype4(in_top[left])     * x1_factor * y1_factor;
    auto tr = ftype4(in_top[right])    * x2_factor * y1_factor;
    auto bl = ftype4(in_bottom[left])  * x1_factor * y2_factor;
    auto br = ftype4(in_bottom[right]) * x2_factor * y2_factor;
    out[int(gid.z) * params.output_size + int(gid.y) * params.output_width + int(gid.x)] = ftype4(tl + tr + bl + br);
}

static inline ftype4 upsample_cubic_interpolation(ftype4 A, ftype4 B, ftype4 C, ftype4 D, float factor) {
    ftype4 a = (B - C) + 0.5f * (B - A) + (D - C) * 0.5f;
    ftype4 b = C - ((B - A) + (B - C)) - (B + D) * 0.5f;
    ftype4 c = (C - A) * 0.5f;
    ftype4 d = B;
    return ((a * factor + b) * factor + c) * factor + d;
}

kernel void upsample_cubic(const device ftype4 *in               [[buffer(0)]],
                           device ftype4 *out                      [[buffer(1)]],
                           constant MetalUpsampleParams &params    [[buffer(2)]],
                           uint3 gid                               [[thread_position_in_grid]]) {
    if (any(gid >= uint3(params.output_width, params.output_height, params.output_slice*params.batch)))
        return;
    
    float u = float(gid.x) / float(params.output_width - 1);
    float v = float(gid.y) / float(params.output_height - 1);
    float x = u * params.input_width - 0.5f;
    float y = v * params.input_height - 0.5f;
    float x_factor = x - floor(x);
    float y_factor = y - floor(y);
    
    int4 xp = int4(int(x) - 1, int(x) + 0, int(x) + 1, int(x) + 2);
    xp = clamp(xp, 0, params.input_width - 1);
    
    int4 yp = int4(int(y) - 1, int(y) + 0, int(y) + 1, int(y) + 2);
    yp = clamp(yp, 0, params.input_height - 1);
    
    auto in_z = in + gid.z * params.input_size;
    ftype4x4 ABCD;
    for (int i = 0; i < 4; i++) {
        auto in_y = in_z + yp[i] * params.input_width;
        ftype4 A = ftype4(in_y[xp[0]]);
        ftype4 B = ftype4(in_y[xp[1]]);
        ftype4 C = ftype4(in_y[xp[2]]);
        ftype4 D = ftype4(in_y[xp[3]]);
        ABCD[i] = upsample_cubic_interpolation(A, B, C, D, x_factor);
    }
    
    auto val = ftype4(upsample_cubic_interpolation(ABCD[0], ABCD[1], ABCD[2], ABCD[3], y_factor));
    out[int(gid.z) * params.output_size + int(gid.y) * params.output_width + int(gid.x)] = val;
}







