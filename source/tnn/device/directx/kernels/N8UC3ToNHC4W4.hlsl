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
    uint image_width_idx  = DTid.x * 4;
    uint image_height_idx = DTid.y;

    uint texture_w;
    uint texture_h;
    Texture_Blob.GetDimensions(texture_w, texture_h);

    if (image_width_idx >= texture_w || image_height_idx >= texture_h) {
        return;
    }

    uint batch_idx  = image_height_idx / height;
    uint height_idx = image_height_idx % height;

    uint buffer_offset = ((batch_idx * height + height_idx) * width + image_width_idx) * 3;

    uint data0 = asuint( BufferIn.Load( (buffer_offset + 0) ) );
    uint data1 = asuint( BufferIn.Load( (buffer_offset + 4) ) );
    uint data2 = asuint( BufferIn.Load( (buffer_offset + 8) ) );

	float b0, g0, r0;
	float b1, g1, r1;
	float b2, g2, r2;
	float b3, g3, r3;

	b1 =  data0 >> 24;
	r0 = (data0 & 0x00ff0000) >> 16;
	g0 = (data0 & 0x0000ff00) >> 8;
	b0 = (data0 & 0x000000ff);

	g2 =  data1 >> 24;
	b2 = (data1 & 0x00ff0000) >> 16;
	r1 = (data1 & 0x0000ff00) >> 8;
	g1 = (data1 & 0x000000ff);

	r3 =  data2 >> 24;
	g3 = (data2 & 0x00ff0000) >> 16;
	b3 = (data2 & 0x0000ff00) >> 8;
	r2 = (data2 & 0x000000ff);


    float4 scale = {scale0,scale1,scale2,scale3};
    float4 bias = {bias0,bias1,bias2,bias3};

    float4 value0 = {b0,g0,r0,0.f};
    float4 value1 = {b1,g1,r1,0.f};
    float4 value2 = {b2,g2,r2,0.f};
    float4 value3 = {b3,g3,r3,0.f};

    value0 = mad(value0, scale, bias);
    value1 = mad(value1, scale, bias);
    value2 = mad(value2, scale, bias);
    value3 = mad(value3, scale, bias);

    uint2 pos0 = {image_width_idx +0, image_height_idx};
    uint2 pos1 = {image_width_idx +1, image_height_idx};
    uint2 pos2 = {image_width_idx +2, image_height_idx};
    uint2 pos3 = {image_width_idx +3, image_height_idx};

    Texture_Blob[pos0] = value0;
    Texture_Blob[pos1] = value1;
    Texture_Blob[pos2] = value2;
    Texture_Blob[pos3] = value3;
}