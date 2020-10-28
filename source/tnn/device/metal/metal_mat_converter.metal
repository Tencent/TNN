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
#include <metal_texture>
#include "tnn/device/metal/acc/metal_common.metal"

using namespace metal;

kernel void mat_converter_texture_n8uc4_crop(
                                             texture2d<half, access::read> src_bgra      [[texture(0)]],
                                             texture2d<half, access::write> dst_bgra     [[texture(1)]],
                                             constant MetalCropParams& parameters        [[buffer(0)]],
                                             ushort2 gid                                 [[thread_position_in_grid]])
{
    auto src_x = gid.x + parameters.top_left_x;
    auto src_y = gid.y + parameters.top_left_y;
    if (src_x >= parameters.width || src_y >= parameters.height || any(gid >= ushort2(parameters.crop_width, parameters.crop_height)))
        return;
    
    auto in = src_bgra.read(uint2(src_x, src_y));
    dst_bgra.write(in, uint2(gid));
}

kernel void mat_converter_texture_n8uc4_resize_nearest(
                                                       texture2d<half, access::sample> src_bgra     [[texture(0)]],
                                                       texture2d<half, access::write> dst_bgra      [[texture(1)]],
                                                       constant MetalResizeParams& parameters       [[buffer(0)]],
                                                       ushort2 gid                                  [[thread_position_in_grid]])
{
    //TODO: the behavior of out-of-bounds reads?
    // clamp_to_edge: coods out-of-bounds will be moved to the edge
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    
    if (any(gid >= ushort2(parameters.resized_width, parameters.resized_height)))
        return;
    
    float x = min(float(gid.x*1.0/parameters.scale_w), float(parameters.width  - 1));
    float y = min(float(gid.y*1.0/parameters.scale_h), float(parameters.height - 1));
    
    auto sampled_color = src_bgra.sample(s, float2(x, y));
    
    dst_bgra.write(sampled_color, uint2(gid));
}

kernel void mat_converter_texture_n8uc4_resize_linear(
                                                      texture2d<half, access::sample> src_bgra      [[texture(0)]],
                                                      texture2d<half, access::write> dst_bgra       [[texture(1)]],
                                                      constant MetalResizeParams& parameters        [[buffer(0)]],
                                                      ushort2 gid                                   [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);
    
    if (any(gid >= ushort2(parameters.resized_width, parameters.resized_height)))
        return;

    float scale_w_inv = float(parameters.width) / float(parameters.resized_width);
    float scale_h_inv = float(parameters.height) / float(parameters.resized_height);
    
    float x = min(float(gid.x * scale_w_inv), float(parameters.width-1));
    float y = min(float(gid.y * scale_h_inv), float(parameters.height-1));

    auto sampled_color = src_bgra.sample(s, float2(x, y));
    dst_bgra.write(sampled_color, uint2(gid));
}

#define CLAMP(v, min, max) \
if (v < min) { \
v = min; \
} else if (v > max) { \
v = max; \
}

float4 GetPixelClamped(texture2d<half, access::read> in [[texture(0)]], uint x, uint y, uint width, uint height) {
    CLAMP(x, 0, width - 1)
    CLAMP(y, 0, height - 1)
    return float4(in.read(uint2(x, y)));
}
#define S_MIN -32768
#define S_MAX 32767
#define SATURATE_CAST_SHORT(x) (half)(min(max(S_MIN, (int)((x)+((x)>=0.f? 0.5f:-0.5f))), S_MAX))

kernel void mat_converter_texture_n8uc4_resize_bilinear(
                                                        texture2d<half, access::read> src_bgra        [[texture(0)]],
                                                        texture2d<half, access::write> dst_bgra       [[texture(1)]],
                                                        constant MetalResizeParams& parameters        [[buffer(0)]],
                                                        ushort2 gid                                   [[thread_position_in_grid]])
{
    if (any(gid >= ushort2(parameters.resized_width, parameters.resized_height)))
        return;
    
    float scale_x_inv = float(parameters.width) / float(parameters.resized_width);
    float scale_y_inv = float(parameters.height) / float(parameters.resized_height);
    
    float x = float(gid.x + 0.5) * scale_x_inv - 0.5;
    float y = float(gid.y + 0.5) * scale_y_inv - 0.5;
    
    int xint = floor(x);
    float xfrac = x - xint;
    if(xint < 0){
        xint = 0;
        xfrac = 0.f;
    }
    if(xint >= parameters.width - 1){
        xint = parameters.width - 2;
        xfrac = 1.f;
    }
    
    int yint = floor(y);
    float yfrac = y - yint;
    if(yint < 0){
        yint = 0;
        yfrac = 0.f;
    }
    if(yint >= parameters.height - 1){
        yint = parameters.height - 2;
        yfrac = 1.f;
    }
    
    float4 p00 = float4(GetPixelClamped(src_bgra, xint + 0, yint + 0, parameters.width, parameters.height))*255.0;
    float4 p10 = float4(GetPixelClamped(src_bgra, xint + 1, yint + 0, parameters.width, parameters.height))*255.0;
    float4 p01 = float4(GetPixelClamped(src_bgra, xint + 0, yint + 1, parameters.width, parameters.height))*255.0;
    float4 p11 = float4(GetPixelClamped(src_bgra, xint + 1, yint + 1, parameters.width, parameters.height))*255.0;

    float x_ef0_ = (1 - xfrac) * 2048;
    float x_ef1_ = xfrac * 2048;
    
    float y_ef0_ = (1 - yfrac) * 2048;
    float y_ef1_ = yfrac * 2048;
    
    float4 col0 = (p00 * x_ef0_ + p10 * x_ef1_) / 16;
    float4 col1 = (p01 * x_ef0_ + p11 * x_ef1_) / 16;
    
    float4 value = ((col0 * y_ef0_)/(1024.0*64.0) + (col1 * y_ef1_)/(1024.0*64.0) + 2.0) / 4.0;
    
    value = value / 255.0;
    
    dst_bgra.write(half4(value), uint2(gid));
}

kernel void mat_converter_texture_n8uc4_resize_bilinear_gather(
                                                               texture2d<half, access::sample> src_bgra      [[texture(0)]],
                                                               texture2d<half, access::write> dst_bgra       [[texture(1)]],
                                                               constant MetalResizeParams& parameters        [[buffer(0)]],
                                                               ushort2 gid                                   [[thread_position_in_grid]])
{
    if(any(gid >= ushort2(parameters.resized_width, parameters.resized_height)))
        return;
    
    //constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);
    //scale_w=1.5, scale_h=1.2
    float x = float(gid.x + 0.5) / parameters.scale_w - 0.5;
    float y = float(gid.y + 0.5) / parameters.scale_h - 0.5;
    
    // the inner order of the gather is: (top-right, bottom-right, bottom-left, top-left)
    half4 val_x = src_bgra.gather(s, float2(x, y), int2(0), component::x);
    half4 val_y = src_bgra.gather(s, float2(x, y), int2(0), component::y);
    half4 val_z = src_bgra.gather(s, float2(x, y), int2(0), component::z);
    half4 val_w = src_bgra.gather(s, float2(x, y), int2(0), component::w);
    
    int left   = floor(x);
    //int right  = min(left + 1, parameters.width - 1);
    int top    = floor(y);
    //int bottom = min(top + 1, parameters.height - 1);
    
    float scale_w2 = x - left;
    float scale_w1 = 1 - scale_w2;
    float scale_h2 = y - top;
    float scale_h1 = 1 - scale_h2;
    
    half4 coef;
    coef[0] = scale_w2*scale_h1;
    coef[1] = scale_w2*scale_h2;
    coef[2] = scale_w1*scale_h2;
    coef[3] = scale_w1*scale_h1;
    
    half4 rst;
    half4 xvec = val_x * coef;
    rst[0] = xvec[0] + xvec[1] + xvec[2] + xvec[3];
    half4 yvec = val_y * coef;
    rst[1] = yvec[0] + yvec[1] + yvec[2] + yvec[3];
    half4 zvec = val_z * coef;
    rst[2] = zvec[0] + zvec[1] + zvec[2] + zvec[3];
    half4 wvec = val_w * coef;
    rst[3] = wvec[0] + wvec[1] + wvec[2] + wvec[3];
    
    dst_bgra.write(rst, uint2(gid));
}

kernel void bgr2gray_n8uc4_nchw_float(
                              texture2d<half, access::read>  src_bgra[[texture(0)]],
                              device float*                       out[[buffer(0)]],
                              constant MetalBGR2GrayParams& parameters [[buffer(1)]],
                              ushort2 gid[[thread_position_in_grid]])
{
    
    if(any(gid >= ushort2(parameters.width, parameters.height)))
        return;
    auto out_offset = gid.y * parameters.width + gid.x;
    
    float4 rgb = float4(src_bgra.read(uint2(gid.xy))) * 255;
    auto bgr = rgb.zyxw;
    
    float rst = float(0);
    rst = bgr[0] * 0.114 + bgr[1] * 0.587 + bgr[2] * 0.299;
    
    out[out_offset] = rst;
}

kernel void copy_nchw_to_cpu(
                             device float* in                       [[buffer(0)]],
                             device float* out                      [[buffer(1)]],
                             constant MetalCopyParams& parameters   [[buffer(2)]],
                             ushort3 gid                            [[thread_position_in_grid]])
{
    if(any(gid >= ushort3(parameters.width, parameters.height, parameters.channel * parameters.batch)))
        return;
    auto offset = gid.z * parameters.width * parameters.height + gid.y * parameters.width + gid.x;
    out[offset] = in[offset];
}

kernel void copy_n8uc4_to_cpu(
                              texture2d<half, access::read> src_bgra[[texture(0)]],
                              device uchar* out[[buffer(0)]],
                              constant MetalCopyParams& parameters [[buffer(1)]],
                              ushort2 gid[[thread_position_in_grid]])
{
    
    if(any(gid >= ushort2(parameters.width, parameters.height)))
        return;
    
    half4 cs = src_bgra.read(uint2(gid.xy));
    half4 shuffle_cs = cs.zyxw;
    uchar4 data = uchar4(shuffle_cs * 255.0);
    
    auto offset = (gid.y * parameters.width + gid.x)*4;
    out[offset + 0] = data[0];
    out[offset + 1] = data[1];
    out[offset + 2] = data[2];
    out[offset + 3] = data[3];
}

kernel void copy_n8uc4_metal_to_n8uc3_cpu(
                                          texture2d<half, access::read> src_bgra    [[texture(0)]],
                                          device uchar* out                         [[buffer(0)]],
                                          constant MetalCopyParams& parameters      [[buffer(1)]],
                                          ushort2 gid                               [[thread_position_in_grid]])
{
    if(any(gid >= ushort2(parameters.width, parameters.height)))
        return;
    
    half4 cs = src_bgra.read(uint2(gid.xy));
    
    half4 shuffle_cs = cs.zyxw;
    uchar4 data = uchar4(shuffle_cs * 255.0);
    
    auto offset = (gid.y * parameters.width + gid.x) * 3;
    out[offset + 0] = data[0];
    out[offset + 1] = data[1];
    out[offset + 2] = data[2];
}

kernel void copy_n8uc3_cpu_to_n8uc4_metal(
                                          device uchar* src                         [[buffer(0)]],
                                          texture2d<half, access::write> dst_bgra   [[texture(0)]],
                                          constant MetalCopyParams& parameters      [[buffer(1)]],
                                          ushort2 gid                               [[thread_position_in_grid]])
{
    if(any(gid >= ushort2(parameters.width, parameters.height)))
        return;
    
    auto offset = (gid.y * parameters.width + gid.x) * 3;
    
    half r = half(src[offset + 0] * 1.0) / 255.0;
    half g = half(src[offset + 1] * 1.0) / 255.0;
    half b = half(src[offset + 2] * 1.0) / 255.0;
    
    half4 data;
    data[0] = r;
    data[1] = g;
    data[2] = b;
    data[3] = 0;
    
    data = data.zyxw;
    
    dst_bgra.write(data, uint2(gid));
}

kernel void mat_converter_texture_n8uc4_warpaffine_linear_const(
                                                                texture2d<half, access::read> src_bgra        [[texture(0)]],
                                                                texture2d<half, access::write> dst_bgra       [[texture(1)]],
                                                                constant MetalWarpAffineParams& parameters    [[buffer(0)]],
                                                                ushort2 gid                                   [[thread_position_in_grid]])
{
    if (any(gid >= ushort2(parameters.resized_width, parameters.resized_height)))
        return;
    
    // fill dst with border value first
    half4 border_value = half4(parameters.border_val / 255.0);
    dst_bgra.write(border_value, uint2(gid));
    
    float x = gid.x * parameters.transform_inv[0][0] + gid.y * parameters.transform_inv[0][1] + parameters.transform_inv[0][2];
    float y = gid.x * parameters.transform_inv[1][0] + gid.y * parameters.transform_inv[1][1] + parameters.transform_inv[1][2];
    
    float4 value = 0;
    
    int xint = floor(x);
    float xfrac = x - xint;
    if(xint < 0){
        xint = 0;
        xfrac = 0.f;
    }
    if(xint >= parameters.width - 1){
        xint = parameters.width - 2;
        xfrac = 1.f;
    }
    
    int yint = floor(y);
    float yfrac = y - yint;
    if(yint < 0){
        yint = 0;
        yfrac = 0.f;
    }
    if(yint >= parameters.height - 1){
        yint = parameters.height - 2;
        yfrac = 1.f;
    }
    if( x >= 0 && x < parameters.width-1 && y>=0 && y< parameters.height-1 ) {
        // normal bilinear sampling
        float4 p00 = float4(GetPixelClamped(src_bgra, xint + 0, yint + 0, parameters.width, parameters.height))*255.0;
        float4 p10 = float4(GetPixelClamped(src_bgra, xint + 1, yint + 0, parameters.width, parameters.height))*255.0;
        float4 p01 = float4(GetPixelClamped(src_bgra, xint + 0, yint + 1, parameters.width, parameters.height))*255.0;
        float4 p11 = float4(GetPixelClamped(src_bgra, xint + 1, yint + 1, parameters.width, parameters.height))*255.0;
        
        float x_ef0_ = (1 - xfrac) * 2048;
        float x_ef1_ = xfrac * 2048;
        
        float y_ef0_ = (1 - yfrac) * 2048;
        float y_ef1_ = yfrac * 2048;
        
        float4 col0 = (p00 * x_ef0_ + p10 * x_ef1_) / 16;
        float4 col1 = (p01 * x_ef0_ + p11 * x_ef1_) / 16;
        
        value = ((col0 * y_ef0_)/(1024.0*64.0) + (col1 * y_ef1_)/(1024.0*64.0) + 2.0) / 4.0;
        value = value / 255.0;
    } else if( x>=-1 && x<=parameters.width-1 && y>=-1 && y<=parameters.height-1 ){
        // partial sampling
        //(x, y)
        bool mask0 = x >= 0 && y >= 0;
        //(x+1, y)
        bool mask1 = x <= (parameters.width - 2) && y >= 0;
        //(x, y+1)
        bool mask2 = x >= 0 && y <= (parameters.height - 2);
        //(x+1, y+1)
        bool mask3 = x <= (parameters.width - 2) && y <= (parameters.height - 2);
        
        float4 p00 = mask0 ? float4(GetPixelClamped(src_bgra, xint + 0, yint + 0, parameters.width, parameters.height))*255.0 : 0;
        float4 p10 = mask1 ? float4(GetPixelClamped(src_bgra, xint + 1, yint + 0, parameters.width, parameters.height))*255.0 : 0;
        float4 p01 = mask2 ? float4(GetPixelClamped(src_bgra, xint + 0, yint + 1, parameters.width, parameters.height))*255.0 : 0;
        float4 p11 = mask3 ? float4(GetPixelClamped(src_bgra, xint + 1, yint + 1, parameters.width, parameters.height))*255.0 : 0;
        
        float x_ef0_ = (1 - xfrac) * 2048;
        float x_ef1_ = xfrac * 2048;
        
        float y_ef0_ = (1 - yfrac) * 2048;
        float y_ef1_ = yfrac * 2048;
        
        float4 col0 = (p00 * x_ef0_ + p10 * x_ef1_) / 16;
        float4 col1 = (p01 * x_ef0_ + p11 * x_ef1_) / 16;
        
        value = ((col0 * y_ef0_)/(1024.0*64.0) + (col1 * y_ef1_)/(1024.0*64.0) + 2.0) / 4.0;
        value = value / 255.0;
    }
    
    dst_bgra.write(half4(value), uint2(gid));
}

kernel void copymakeborder_n8uc4_constant(
                              texture2d<half, access::read> src_bgra[[texture(0)]],
                              texture2d<half, access::write>dst_bgra[[texture(1)]],
                              constant MetalCopyMakeBorderParam& parameters [[buffer(0)]],
                              ushort2 gid[[thread_position_in_grid]])
{
    uint2 dst_size;
    int dst_height = parameters.height + parameters.top + parameters.bottom;
    int dst_width  = parameters.width + parameters.left + parameters.right;
    dst_size.x = dst_width;
    dst_size.y = dst_height;
    if(any(gid >= (ushort2)dst_size))
        return;

    int2 in_loc = int2(gid.x-parameters.left, gid.y-parameters.top);
    half4 value = half4(parameters.border_val / 255.0);
    if(in_loc.x >= 0 && in_loc.x < parameters.width && in_loc.y >= 0 && in_loc.y < parameters.height)
        value = src_bgra.read(uint2(in_loc));

    dst_bgra.write(value, uint2(gid));
}

kernel void copymakeborder_nchw_constant(
                                device float* in                       [[buffer(0)]],
                                device float* out                      [[buffer(1)]],
                                constant MetalCopyMakeBorderParam& parameters   [[buffer(2)]],
                                ushort3 gid                            [[thread_position_in_grid]])
{
    uint3 dst_size;
    int dst_height = parameters.height + parameters.top + parameters.bottom;
    int dst_width  = parameters.width + parameters.left + parameters.right;
    int dst_slice  = parameters.batch * parameters.channel;
    dst_size.x = dst_width;
    dst_size.y = dst_height;
    dst_size.z = dst_slice;
    if(any(gid >= ushort3(dst_size)))
        return;

    auto dst_offset = gid.z * dst_height * dst_width + gid.y * dst_width + gid.x;

    auto src_h = gid.y - parameters.top;
    auto src_w = gid.x - parameters.left;
    float value = parameters.border_val;
    if (src_h >= 0 && src_h < parameters.height && src_w >= 0 && src_w < parameters.width) {
        auto src_offset = gid.z * parameters.width * parameters.height + src_h * parameters.width + src_w;
        value = in[src_offset];
    }

    out[dst_offset] = value;
}
