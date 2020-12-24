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

#pragma mark - texture -> buffer
kernel void image_converter_texture_bgra8888_2_buffer_nc4hw4(
       texture2d<half, access::read> src_bgra      [[texture(0)]],
       device ftype4 *dst                           [[buffer(0)]],
       constant MetalImageConverterParams& params  [[buffer(1)]],
       ushort2 gid                                 [[thread_position_in_grid]])
{
    //NOTE:
    //output of func read is rgba data, no matter the pixel format of texture is
    //MTLPixelFormatRGBA8Unorm or MTLPixelFormatBGRA8Unorm
    if (any(gid >= ushort2(params.width, params.height)))
        return;
    float4 in = float4(src_bgra.read(uint2(gid)));
    in = !params.bgra_to_rgba ? in.zyxw : in;
    auto out = dst + (int)gid.y * params.width + (int)gid.x;
    
    *out = ftype4(in*float4(params.scale_x, params.scale_y, params.scale_z, params.scale_w) + float4(params.bias_x, params.bias_y, params.bias_z, params.bias_w));
}

kernel void image_converter_texture_bgra8888_2_buffer_nchw_f(
      texture2d<half, access::read> src_bgra      [[texture(0)]],
      device float *dst                           [[buffer(0)]],
      constant MetalImageConverterParams& params  [[buffer(1)]],
      ushort2 gid                                 [[thread_position_in_grid]])
{
    if (any(gid >= ushort2(params.width, params.height)))
        return;
    float4 in = float4(src_bgra.read(uint2(gid)));
    in = !params.bgra_to_rgba ? in.zyxw : in;
    const int offset = (int)gid.y * params.width + (int)gid.x;
    const int channel_size = params.height * params.width;
    in = in*float4(params.scale_x, params.scale_y, params.scale_z, params.scale_w) + float4(params.bias_x, params.bias_y, params.bias_z, params.bias_w);
    dst[offset]                = in.x;
    dst[offset+channel_size]   = in.y;
    dst[offset+channel_size*2] = in.z;
    dst[offset+channel_size*3] = in.w;
}

#pragma mark - buffer -> texture
kernel void image_converter_buffer_nc4hw4_2_texture_bgra8888(
        texture2d<half, access::write> dst_bgra     [[texture(0)]],
        const device ftype4 *src                     [[buffer(0)]],
        constant MetalImageConverterParams& params  [[buffer(1)]],
        ushort2 gid                                 [[thread_position_in_grid]])
{
    if (any(gid >= ushort2(params.width, params.height)))
        return;
    
    float4 in  = float4(src[(int)gid.y * params.width + (int)gid.x]);
    in = in*float4(params.scale_x, params.scale_y, params.scale_z, params.scale_w) + float4(params.bias_x, params.bias_y, params.bias_z, params.bias_w);
    in = !params.bgra_to_rgba ? in.zyxw : in;
    dst_bgra.write(half4(in), uint2(gid));
}


kernel void image_converter_buffer_nchw_f_2_texture_bgra8888(
       texture2d<half, access::write> dst_bgra    [[texture(0)]],
       const device float *src                    [[buffer(0)]],
       constant MetalImageConverterParams& params      [[buffer(1)]],
       ushort2 gid                                 [[thread_position_in_grid]])
{
    if (any(gid >= ushort2(params.width, params.height)))
        return;
    
    const int offset = (int)gid.y * params.width + (int)gid.x;
    const int channel_size = params.height * params.width;
    
    float4 in  = float4(src[offset], src[offset+channel_size], src[offset+channel_size*2], src[offset+channel_size*3]);
    in = in*float4(params.scale_x, params.scale_y, params.scale_z, params.scale_w) + float4(params.bias_x, params.bias_y, params.bias_z, params.bias_w);
    
    in = !params.bgra_to_rgba ? in.zyxw : in;
    dst_bgra.write(half4(in), uint2(gid));
}

#pragma mark - buffer <-> buffer
template<typename SrcType, typename SrcType4, typename DstType, typename DstType4>
static inline void data_converter_nc4hw4_2_nchw(device DstType *dst,
                                                const device SrcType4 *src,
                                                constant MetalImageConverterParams& params,
                                                uint3 gid)
{
    if (any(gid >= uint3(params.size, params.slice, params.batch)))
        return;
    
    const int index_in =  (int)gid.z*params.slice*params.size + (int)gid.y*params.size + (int)gid.x;
    
    int channel_out = gid.y*4;
    int index_out = ((int)gid.z*params.channel + channel_out)*params.size + (int)gid.x;
    
    float4 in_data  = float4(src[index_in]);
    in_data = in_data*float4(params.scale_x, params.scale_y, params.scale_z, params.scale_w) + float4(params.bias_x, params.bias_y, params.bias_z, params.bias_w);
    
    auto out_data = DstType4(in_data);
    
    dst[index_out]                = out_data.x;
    if (channel_out + 1 < params.channel) {
        dst[index_out+params.size]   = out_data.y;
    }
    if (channel_out + 2 < params.channel) {
        dst[index_out+params.size*2] = out_data.z;
    }
    if (channel_out + 3 < params.channel) {
        dst[index_out+params.size*3] = out_data.w;
    }
}

template<typename SrcType, typename SrcType4, typename DstType, typename DstType4>
static inline void data_converter_nc4hw4_2_nchw_v2(device DstType *dst,
                                                const device SrcType4 *src,
                                                constant MetalImageConverterParams& params,
                                                const device DstType *scale,
                                                const device DstType* bias,
                                                uint3 gid)
{
    if (any(gid >= uint3(params.size, params.slice, params.batch)))
        return;

    const int index_in =  (int)gid.z*params.slice*params.size + (int)gid.y*params.size + (int)gid.x;

    int channel_out = gid.y*4;
    int index_out = ((int)gid.z*params.channel + channel_out)*params.size + (int)gid.x;

    float4 scale_c = float4(scale[channel_out], 0, 0, 0);
    float4 bias_c  = float4(bias[channel_out], 0, 0, 0);
    if (channel_out + 1 < params.channel) {
        scale_c.y = scale[channel_out + 1];
        bias_c.y = bias[channel_out + 1];
    }
    if (channel_out + 2 < params.channel) {
        scale_c.z = scale[channel_out + 2];
        bias_c.z = bias[channel_out + 2];
    }
    if (channel_out + 3 < params.channel) {
        scale_c.w = scale[channel_out + 3];
        bias_c.w = bias[channel_out + 3];
    }

    float4 in_data  = float4(src[index_in]);
    in_data = in_data * scale_c + bias_c;

    auto out_data = DstType4(in_data);

    dst[index_out]                = out_data.x;
    if (channel_out + 1 < params.channel) {
        dst[index_out+params.size]   = out_data.y;
    }
    if (channel_out + 2 < params.channel) {
        dst[index_out+params.size*2] = out_data.z;
    }
    if (channel_out + 3 < params.channel) {
        dst[index_out+params.size*3] = out_data.w;
    }
}
    
kernel void data_converter_nc4hw4_2_nchw_float(
                                             device float *dst                             [[buffer(0)]],
                                             const device ftype4 *src                   [[buffer(1)]],
                                             constant MetalImageConverterParams& params      [[buffer(2)]],
                                             uint3 gid                                 [[thread_position_in_grid]])
{
    data_converter_nc4hw4_2_nchw<ftype, ftype4, float, float4>(dst, src, params, gid);
}

kernel void data_converter_nc4hw4_2_nchw_float_v2(
                                             device float *dst                             [[buffer(0)]],
                                             const device ftype4 *src                   [[buffer(1)]],
                                             constant MetalImageConverterParams& params      [[buffer(2)]],
                                             const device float *scale                  [[buffer(3)]],
                                             const device float *bias                   [[buffer(4)]],
                                             uint3 gid                                 [[thread_position_in_grid]])
{
    data_converter_nc4hw4_2_nchw_v2<ftype, ftype4, float, float4>(dst, src, params, scale, bias, gid);
}

template<typename SrcType, typename SrcType4, typename DstType, typename DstType4>
static inline void data_converter_nchw_2_nc4hw4(device DstType4 *dst,
                                                const device SrcType *src,
                                                constant MetalImageConverterParams& params,
                                                uint3 gid)
{
    if (any(gid >= uint3(params.size, params.slice, params.batch)))
        return;
    
    int channel_in= gid.y*4;
    int index_in = ((int)gid.z*params.channel + channel_in)*params.size + (int)gid.x;
    
    const int index_out =  (int)gid.z*params.slice*params.size + (int)gid.y * params.size + (int)gid.x;
    
    float4 in_data  = float4(Zero4);
    float4 in_mask = float4(1, 0, 0, 0);
    
    in_data.x = src[index_in];
    if (channel_in + 1 < params.channel) {
        in_data.y = src[index_in + params.size];
        in_mask.y = 1;
    }
    if (channel_in + 2 < params.channel) {
        in_data.z = src[index_in + params.size*2];
        in_mask.z = 1;
    }
    if (channel_in + 3 < params.channel) {
        in_data.w = src[index_in + params.size*3];
        in_mask.w = 1;
    }
    
    in_data = in_data*float4(params.scale_x, params.scale_y, params.scale_z, params.scale_w) + float4(params.bias_x, params.bias_y, params.bias_z, params.bias_w);
    in_data *= in_mask;
    
    dst[index_out] = DstType4(in_data);

}

template<typename SrcType, typename SrcType4, typename DstType, typename DstType4>
static inline void data_converter_nchw_2_nc4hw4_v2(device DstType4 *dst,
                                                const device SrcType *src,
                                                constant MetalImageConverterParams& params,
                                                const device SrcType *scale,
                                                const device SrcType* bias,
                                                uint3 gid)
{
    if (any(gid >= uint3(params.size, params.slice, params.batch)))
        return;

    int channel_in= gid.y*4;
    int index_in = ((int)gid.z*params.channel + channel_in)*params.size + (int)gid.x;

    const int index_out =  (int)gid.z*params.slice*params.size + (int)gid.y * params.size + (int)gid.x;

    float4 in_data  = float4(Zero4);
    float4 scale_c  = float4(Zero4);
    float4 bias_c   = float4(Zero4);

    in_data.x = src[index_in];
    scale_c.x = scale[channel_in];
    bias_c.x  = bias[channel_in];
    if (channel_in + 1 < params.channel) {
        in_data.y = src[index_in + params.size];
        scale_c.y = scale[channel_in + 1];
        bias_c.y  = bias[channel_in + 1];
    }
    if (channel_in + 2 < params.channel) {
        in_data.z = src[index_in + params.size*2];
        scale_c.z = scale[channel_in + 2];
        bias_c.z  = bias[channel_in + 2];
    }
    if (channel_in + 3 < params.channel) {
        in_data.w = src[index_in + params.size*3];
        scale_c.w = scale[channel_in + 3];
        bias_c.w  = bias[channel_in + 3];
    }

    in_data = in_data * scale_c + bias_c;

    dst[index_out] = DstType4(in_data);
}

kernel void data_converter_nchw_2_nc4hw4_float(
                                             device ftype4 *dst                             [[buffer(0)]],
                                             const device float *src                   [[buffer(1)]],
                                             constant MetalImageConverterParams& params      [[buffer(2)]],
                                             uint3 gid                                 [[thread_position_in_grid]])
{
    data_converter_nchw_2_nc4hw4<float, float4, ftype, ftype4>(dst, src, params, gid);
}

kernel void data_converter_nchw_2_nc4hw4_float_v2(
                                             device ftype4 *dst                             [[buffer(0)]],
                                             const device float *src                   [[buffer(1)]],
                                             constant MetalImageConverterParams& params      [[buffer(2)]],
                                             const device float *scale                  [[buffer(3)]],
                                             const device float *bias                   [[buffer(4)]],
                                             uint3 gid                                 [[thread_position_in_grid]])
{
    data_converter_nchw_2_nc4hw4_v2<float, float4, ftype, ftype4>(dst, src, params, scale, bias, gid);
}
