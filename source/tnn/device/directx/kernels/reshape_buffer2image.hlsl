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
    // input dimension
    vector<uint, 4> id;

    // output dimension
    vector<uint, 4> od;
};

ByteAddressBuffer BufferIn : register(t0);
RWTexture2D<float4> Texture_Blob : register(u0);

[numthreads(4, 4, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    uint batch = od[0];
    uint channel = od[1];
    uint height = od[2];
    uint width = od[3];

    uint image_width_idx  = DTid.x;
    uint image_height_idx = DTid.y;

    uint texture_w;
    uint texture_h;
    Texture_Blob.GetDimensions(texture_w, texture_h);
    if (image_width_idx >= texture_w || image_height_idx >= texture_h) {
        return;
    }

    uint batch_idx     = image_height_idx / height;
    uint height_idx    = image_height_idx % height;
    uint width_idx     = image_width_idx % width;
    uint channel_4_idx = (image_width_idx / width) * 4;
    uint buffer_offset = ((batch_idx * channel + channel_4_idx) * height + height_idx) * width + width_idx;

    uint remain_channel    = channel - channel_4_idx;
    uint height_width_size = height * width;
    float4 output_values   = {0,0,0,0};

    uint offset     = buffer_offset;

    if (remain_channel >= 4) {
        output_values.x = asfloat( BufferIn.Load( offset*4 ) );
        offset += height_width_size;
        output_values.y = asfloat( BufferIn.Load( offset*4 ) );
        offset += height_width_size;
        output_values.z = asfloat( BufferIn.Load( offset*4 ) );
        offset += height_width_size;
        output_values.w = asfloat( BufferIn.Load( offset*4 ) );

        Texture_Blob[DTid.xy] = output_values;
    } else if (remain_channel == 3) {
        output_values.x = asfloat( BufferIn.Load( offset*4 ) );
        offset += height_width_size;
        output_values.y = asfloat( BufferIn.Load( offset*4 ) );
        offset += height_width_size;
        output_values.z = asfloat( BufferIn.Load( offset*4 ) );
        output_values.w = 0;

        Texture_Blob[DTid.xy] = output_values;
    } else if (remain_channel == 2) {
        output_values.x = asfloat( BufferIn.Load( offset*4 ) );
        offset += height_width_size;
        output_values.y = asfloat( BufferIn.Load( offset*4 ) );
        output_values.z = 0;
        output_values.w = 0;

        Texture_Blob[DTid.xy] = output_values;
    } else if (remain_channel == 1) {
        output_values.x = asfloat( BufferIn.Load( offset*4 ) );
        output_values.y = 0;
        output_values.z = 0;
        output_values.w = 0;

       Texture_Blob[DTid.xy] = output_values;
    }

}