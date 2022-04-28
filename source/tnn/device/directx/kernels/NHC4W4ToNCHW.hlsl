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

#define UP_DIV(A, B) (((A) + (B) - 1) / B)

cbuffer InputCBBuffer : register(b0)
{
     float scale0;
     float scale1;
     float scale2;
     float scale3;

     float bias0;
     float bias1;
     float bias2;
     float bias3;

     int batch;
     int channel;
     int height;
     int width;

};

Texture2D<float4> Texture_Blob : register(t0);
RWByteAddressBuffer BufferOut : register(u0);

[numthreads(4, 4, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    uint image_width_idx  = DTid.x;
    uint image_height_idx = DTid.y;

//     uint texture_w;
//     uint texture_h;
//     Texture_Blob.GetDimensions(texture_w, texture_h);
    if (image_width_idx >= UP_DIV(channel, 4)*width || image_height_idx >=  batch*height ) {
        return;
    }

    uint batch_idx     = image_height_idx / height;
    uint height_idx    = image_height_idx % height;
    uint width_idx     = image_width_idx % width;
    uint channel_4_idx = (image_width_idx / width) * 4;
    uint buffer_offset = ((batch_idx * channel + channel_4_idx) * height + height_idx) * width + width_idx;

    float4 values = Texture_Blob[DTid.xy];

    uint height_width_size = height * width;
    uint remain_channel = channel - channel_4_idx;

    float4 scale_data   = {scale0,scale1,scale2,scale3};
    float4 bias_data    = {bias0,bias1,bias2,bias3};
    uint offset     = buffer_offset;

    if (remain_channel >= 4) {
        values = mad(values, scale_data, bias_data);

        BufferOut.Store( offset*4, asuint(values.x) );
        offset += height_width_size;
        BufferOut.Store( offset*4, asuint(values.y) );
        offset += height_width_size;
        BufferOut.Store( offset*4, asuint(values.z) );
        offset += height_width_size;
        BufferOut.Store( offset*4, asuint(values.w) );
    } else if (remain_channel == 3) {
        values = mad(values, scale_data, bias_data);

        BufferOut.Store( offset*4, asuint(values.x) );
        offset += height_width_size;
        BufferOut.Store( offset*4, asuint(values.y) );
        offset += height_width_size;
        BufferOut.Store( offset*4, asuint(values.z) );
    } else if (remain_channel == 2) {
        values = mad(values, scale_data, bias_data);

        BufferOut.Store( offset*4, asuint(values.x) );
        offset += height_width_size;
        BufferOut.Store( offset*4, asuint(values.y) );
    } else if (remain_channel == 1) {
        values = mad(values, scale_data, bias_data);

        BufferOut.Store( offset*4, asuint(values.x) );
    }


}