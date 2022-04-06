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

ByteAddressBuffer BufferIn : register(t0);
RWTexture2D<float4> Texture_Blob : register(u0);

[numthreads(1, 1, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    uint image_width_idx  = DTid.x;
    uint image_height_idx = DTid.y;

    uint texture_w;
    uint texture_h;
    Texture_Blob.GetDimensions(texture_w, texture_h);

    if (image_width_idx >= texture_w || image_height_idx >= texture_h) {
        return;
    }

    uint batch_idx  = image_height_idx / height;
    uint height_idx = image_height_idx % height;

    uint buffer_offset = ((batch_idx * height + height_idx) * width + image_width_idx) * 4;

    uint data = asuint( BufferIn.Load( buffer_offset ) );
	float b, g, r, a;

    if (channel == 3) {
        a = 0.f;
    } else {
    	a =  data >> 24;
    }
    r = (data & 0x00ff0000) >> 16;
    g = (data & 0x0000ff00) >> 8;
    b = (data & 0x000000ff);

    float4 values = {b,g,r,a};
    float4 scale = {scale0,scale1,scale2,scale3};
    float4 bias = {bias0,bias1,bias2,bias3};

    values = values * scale + bias;
    Texture_Blob[DTid.xy] = values;
}


