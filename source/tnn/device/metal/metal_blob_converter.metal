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

static inline uchar convert_uchar_saturate(const ftype val) {
    return val <= 0 ? uchar(0) : (val >= 255 ? uchar(255) : uchar(val));
}

#pragma mark - buffer <-> buffer
kernel void image_converter_buffer_nc4hw4_2_buffer_bgra(
      device uchar  *dst                           [[buffer(0)]],
      const device ftype4 *src                     [[buffer(1)]],
      constant MetalImageConverterParams& params   [[buffer(2)]],
      ushort2 gid                                  [[thread_position_in_grid]])
{
    if (any(gid >= ushort2(params.width, params.height)))
        return;

    const int offset = (int)gid.y * params.width + (int)gid.x;

    float4 in = float4(src[offset]);
    
    in = in*float4(params.scale_x, params.scale_y, params.scale_z, params.scale_w) + float4(params.bias_x, params.bias_y, params.bias_z, params.bias_w);
    in = params.bgra_to_rgba ? in.zyxw : in;

    dst[offset*4 + 0] = convert_uchar_saturate(in.x);
    dst[offset*4 + 1] = convert_uchar_saturate(in.y);
    dst[offset*4 + 2] = convert_uchar_saturate(in.z);
    dst[offset*4 + 3] = convert_uchar_saturate(in.w);
}

kernel void image_converter_buffer_nc4hw4_2_buffer_bgr(
      device uchar  *dst                           [[buffer(0)]],
      const device ftype4 *src                     [[buffer(1)]],
      constant MetalImageConverterParams& params   [[buffer(2)]],
      ushort2 gid                                  [[thread_position_in_grid]])
{
    if (any(gid >= ushort2(params.width, params.height)))
        return;

    const int offset = (int)gid.y * params.width + (int)gid.x;

    float3 in = float3(src[offset].xyz);
    
    in = in*float3(params.scale_x, params.scale_y, params.scale_z) + float3(params.bias_x, params.bias_y, params.bias_z);
    in = params.bgra_to_rgba ? in.zyx : in;

    dst[offset*3 + 0] = convert_uchar_saturate(in.x);
    dst[offset*3 + 1] = convert_uchar_saturate(in.y);
    dst[offset*3 + 2] = convert_uchar_saturate(in.z);
}

kernel void image_converter_buffer_bgr_2_buffer_nc4hw4(
      device ftype4 *dst                          [[buffer(0)]],
      const device uchar  *src                    [[buffer(1)]],
      constant MetalImageConverterParams& params  [[buffer(2)]],
      ushort2 gid                                 [[thread_position_in_grid]])
{
    if (any(gid >= ushort2(params.width, params.height)))
        return;

    const int offset = (int)gid.y * params.width + (int)gid.x;

    float3 in = float3(src[offset*3], src[offset*3 + 1], src[offset*3 + 2]);
    in = params.bgra_to_rgba ? in.zyx : in;
    
    in = in*float3(params.scale_x, params.scale_y, params.scale_z) + float3(params.bias_x, params.bias_y, params.bias_z);
    
    ftype4 val  = ftype4(in.x, in.y, in.z, 0.f);
    dst[offset] = val;
}

kernel void image_converter_buffer_bgra_2_buffer_nc4hw4(
      device ftype4 *dst                          [[buffer(0)]],
      const device uchar  *src                    [[buffer(1)]],
      constant MetalImageConverterParams& params  [[buffer(2)]],
      ushort2 gid                                 [[thread_position_in_grid]])
{
    if (any(gid >= ushort2(params.width, params.height)))
        return;

    const int offset = (int)gid.y * params.width + (int)gid.x;

    float4 in = float4(src[offset*4], src[offset*4 + 1], src[offset*4 + 2], src[offset*4 + 3]);
    in = params.bgra_to_rgba ? in.zyxw : in;
    
    in = in*float4(params.scale_x, params.scale_y, params.scale_z, params.scale_w) + float4(params.bias_x, params.bias_y, params.bias_z, params.bias_w);
    dst[offset] = ftype4(in.x);
}

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

template<typename SrcType, typename SrcType4, typename DstType, typename DstType4, bool DoScale=true>
static inline void data_converter_nc4hw4_2_nchw_v2(device DstType *dst,
                                                const device SrcType4 *src,
                                                constant MetalImageConverterParams& params,
                                                const device float *scale,
                                                const device float * bias,
                                                uint3 gid)
{
    if (any(gid >= uint3(params.size, params.slice, params.batch)))
        return;

    const int index_in =  (int)gid.z*params.slice*params.size + (int)gid.y*params.size + (int)gid.x;

    int channel_out = gid.y*4;
    int index_out = ((int)gid.z*params.channel + channel_out)*params.size + (int)gid.x;

    float4 scale_c = float4(One4);
    float4 bias_c  = float4(Zero4);
    if (DoScale) {
        scale_c = float4(scale[channel_out], 0, 0, 0);
        bias_c  = float4(bias[channel_out], 0, 0, 0);
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
    }

    SrcType4 in_data  = src[index_in];
    auto out_data = DstType4(in_data);
    if (DoScale) {
        float4 value = float4(in_data) * scale_c + bias_c;
        out_data = DstType4(value);
    }

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

kernel void data_converter_nc4hw4_2_nchw_half_v2(
                                             device half *dst                             [[buffer(0)]],
                                             const device ftype4 *src                   [[buffer(1)]],
                                             constant MetalImageConverterParams& params      [[buffer(2)]],
                                             const device float *scale                  [[buffer(3)]],
                                             const device float *bias                   [[buffer(4)]],
                                             uint3 gid                                 [[thread_position_in_grid]])
{
    data_converter_nc4hw4_2_nchw_v2<ftype, ftype4, half, half4>(dst, src, params, scale, bias, gid);
}

kernel void data_converter_nc4hw4_2_nchw_int32_v2(
                                             device int *dst                             [[buffer(0)]],
                                             const device int4 *src                   [[buffer(1)]],
                                             constant MetalImageConverterParams& params      [[buffer(2)]],
                                             const device float *scale                  [[buffer(3)]],
                                             const device float *bias                   [[buffer(4)]],
                                             uint3 gid                                 [[thread_position_in_grid]])
{
    data_converter_nc4hw4_2_nchw_v2<int, int4, int, int4, false>(dst, src, params, scale, bias, gid);
}

kernel void data_converter_nc4hw4_2_nchw_int322float_v2(
                                             device float *dst                             [[buffer(0)]],
                                             const device int4 *src                   [[buffer(1)]],
                                             constant MetalImageConverterParams& params      [[buffer(2)]],
                                             const device float *scale                  [[buffer(3)]],
                                             const device float *bias                   [[buffer(4)]],
                                             uint3 gid                                 [[thread_position_in_grid]])
{
    data_converter_nc4hw4_2_nchw_v2<int, int4, float, float4>(dst, src, params, scale, bias, gid);
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

template<typename SrcType, typename SrcType4, typename DstType, typename DstType4, bool DoScale=true>
static inline void data_converter_nchw_2_nc4hw4_v2(device DstType4 *dst,
                                                const device SrcType *src,
                                                constant MetalImageConverterParams& params,
                                                const device float *scale,
                                                const device float *bias,
                                                uint3 gid)
{
    if (any(gid >= uint3(params.size, params.slice, params.batch)))
        return;

    int channel_in= gid.y*4;
    int index_in = ((int)gid.z*params.channel + channel_in)*params.size + (int)gid.x;

    const int index_out =  (int)gid.z*params.slice*params.size + (int)gid.y * params.size + (int)gid.x;

    ftype4 in_data  = ftype4(Zero4);
    float4 scale_c  = float4(One4);
    float4 bias_c   = float4(Zero4);

    in_data.x = src[index_in];
    if (DoScale == true) {
        scale_c.x = scale[channel_in];
        bias_c.x  = bias[channel_in];
    }
    if (channel_in + 1 < params.channel) {
        in_data.y = src[index_in + params.size];
        if (DoScale) {
            scale_c.y = scale[channel_in + 1];
            bias_c.y  = bias[channel_in + 1];
        }
    }
    if (channel_in + 2 < params.channel) {
        in_data.z = src[index_in + params.size*2];
        if (DoScale) {
            scale_c.z = scale[channel_in + 2];
            bias_c.z  = bias[channel_in + 2];
        }
    }
    if (channel_in + 3 < params.channel) {
        in_data.w = src[index_in + params.size*3];
        if (DoScale) {
            scale_c.w = scale[channel_in + 3];
            bias_c.w  = bias[channel_in + 3];
        }
    }

    ftype4 result = in_data;
    if (DoScale) {
        result = ftype4(float4(in_data) * scale_c + bias_c);
    }

    dst[index_out] = DstType4(result);
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

kernel void data_converter_nchw_2_nc4hw4_half_v2(
                                             device ftype4 *dst                             [[buffer(0)]],
                                             const device half *src                   [[buffer(1)]],
                                             constant MetalImageConverterParams& params      [[buffer(2)]],
                                             const device float *scale                  [[buffer(3)]],
                                             const device float *bias                   [[buffer(4)]],
                                             uint3 gid                                 [[thread_position_in_grid]])
{
    data_converter_nchw_2_nc4hw4_v2<half, half4, ftype, ftype4>(dst, src, params, scale, bias, gid);
}

kernel void data_converter_nchw_2_nc4hw4_int32_v2(
                                             device int4 *dst                             [[buffer(0)]],
                                             const device int *src                   [[buffer(1)]],
                                             constant MetalImageConverterParams& params      [[buffer(2)]],
                                             const device float *scale                  [[buffer(3)]],
                                             const device float *bias                   [[buffer(4)]],
                                             uint3 gid                                 [[thread_position_in_grid]])
{
    data_converter_nchw_2_nc4hw4_v2<int, int4, int, int4, false>(dst, src, params, scale, bias, gid);
}

kernel void data_converter_nchw_2_nc4hw4_ftype_identity(
                                             device ftype4 *dst                             [[buffer(0)]],
                                             const device ftype *src                   [[buffer(1)]],
                                             constant MetalImageConverterParams& params      [[buffer(2)]],
                                             const device float *scale                  [[buffer(3)]],
                                             const device float *bias                   [[buffer(4)]],
                                             uint3 gid                                 [[thread_position_in_grid]])
{
    data_converter_nchw_2_nc4hw4_v2<ftype, ftype4, ftype, ftype4, false>(dst, src, params, nullptr, nullptr, gid);
}

template<typename SrcType, typename DstType>
static inline void data_converter_nchw_copy_type(device DstType *dst,
                                                const device SrcType *src,
                                            constant MetalImageConverterParams& params,
                                                uint3 gid)
{
    if (any(gid >= uint3(params.size, params.channel, params.batch)))
        return;

    int index = ((int)gid.z*params.channel + (int)gid.y)*params.size + (int)gid.x;
    dst[index] = DstType(src[index]);
}

kernel void data_converter_nchw_ftype2float(device float *dst      [[buffer(0)]],
                                         const device ftype *src  [[buffer(1)]],
                                         constant MetalImageConverterParams& params      [[buffer(2)]],
                                         const device float *scale                  [[buffer(3)]],
                                         const device float *bias                   [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]])
{
    data_converter_nchw_copy_type<ftype, float>(dst, src, params, gid);
}

kernel void data_converter_nchw_float2ftype(device ftype *dst      [[buffer(0)]],
                                         const device float *src  [[buffer(1)]],
                                         constant MetalImageConverterParams& params      [[buffer(2)]],
                                         const device float *scale                  [[buffer(3)]],
                                         const device float *bias                   [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]])
{
    data_converter_nchw_copy_type<float, ftype>(dst, src, params, gid);
}

kernel void data_converter_nchw_ftype2half(device half *dst      [[buffer(0)]],
                                         const device ftype *src  [[buffer(1)]],
                                         constant MetalImageConverterParams& params      [[buffer(2)]],
                                         const device float *scale                  [[buffer(3)]],
                                         const device float *bias                   [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]])
{
    data_converter_nchw_copy_type<ftype, half>(dst, src, params, gid);
}

kernel void data_converter_nchw_half2ftype(device ftype *dst      [[buffer(0)]],
                                         const device half *src  [[buffer(1)]],
                                         constant MetalImageConverterParams& params      [[buffer(2)]],
                                         const device float *scale                  [[buffer(3)]],
                                         const device float *bias                   [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]])
{
    data_converter_nchw_copy_type<half, ftype>(dst, src, params, gid);
}

kernel void data_converter_nchw(device ftype *dst      [[buffer(0)]],
                                         const device ftype *src  [[buffer(1)]],
                                         constant MetalImageConverterParams& params      [[buffer(2)]],
                                         const device float *scale                  [[buffer(3)]],
                                         const device float *bias                   [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]])
{
    data_converter_nchw_copy_type<ftype, ftype>(dst, src, params, gid);
}

kernel void data_converter_nchw_int(device int *dst      [[buffer(0)]],
                                         const device int *src  [[buffer(1)]],
                                         constant MetalImageConverterParams& params      [[buffer(2)]],
                                         const device float *scale                  [[buffer(3)]],
                                         const device float *bias                   [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]])
{
    data_converter_nchw_copy_type<int, int>(dst, src, params, gid);
}