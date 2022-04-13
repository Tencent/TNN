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

cbuffer Shapes: register( b0 )
{
    // NB
    // in a constant buffer, each element of an array must start on a 4-float boundary.
    // so we choose float4 for the ease of alignment with cpp

    // filter a dimension oihw
    // shape[0] shape[1] shape[2] shape[3]
    //   o        i        h        w
    //   x        y        z        w
    vector<uint, 4> shape;

};

//from buffer(oihw) to image [w,h]=(ic oc4, oc/4 h w)
ByteAddressBuffer input : register(t0);
RWTexture2D<float4> output : register(u0);

[numthreads(1, 1, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    int image_width_idx  = DTid.x; // ic
    int image_height_idx = DTid.y; // oc/4 h w

    if (image_width_idx >= shape[1] || image_height_idx >= UP_DIV(shape[0], 4)*shape[2]*shape[3]) {
        return;
    }

    int output_channel = shape[0];
    int2 kernel_wh = {shape[3],shape[2]};
    int ic_h_w_size = shape[1]*shape[2]*shape[3];
    int height_width_size = shape[2]*shape[3];

    int input_channel_4_idx  = image_width_idx;
    int output_channel_4_idx = (image_height_idx / height_width_size) * 4;
    int height_width_idx     = image_height_idx % height_width_size;
    int buffer_height_idx    = height_width_idx / kernel_wh.y;
    int buffer_width_idx     = height_width_idx % kernel_wh.y;

    int buffer_offset = output_channel_4_idx * ic_h_w_size + input_channel_4_idx * height_width_size +
                              buffer_height_idx * kernel_wh.y + buffer_width_idx;

    float4 output_values = 0;
    if (output_channel_4_idx < output_channel) {
        int remain_channel = output_channel - output_channel_4_idx;
        if (remain_channel >= 4) {
            int offset      = buffer_offset;
            output_values.x = asfloat( input.Load( offset*4 ) );
            offset          = mad(1, ic_h_w_size, offset);
            output_values.y = asfloat( input.Load( offset*4 ) );
            offset += ic_h_w_size;
            output_values.z = asfloat( input.Load( offset*4 ) );
            offset += ic_h_w_size;
            output_values.w = asfloat( input.Load( offset*4 ) );

        } else if (remain_channel == 3) {
            int offset      = buffer_offset;
            output_values.x = asfloat( input.Load( offset*4 ) );
            offset          = mad(1, ic_h_w_size, offset);
            output_values.y = asfloat( input.Load( offset*4 ) );
            offset += ic_h_w_size;
            output_values.z = asfloat( input.Load( offset*4 ) );

        } else if (remain_channel == 2) {
            int offset      = buffer_offset;
            output_values.x = asfloat( input.Load( offset*4 ) );
            offset          = mad(1, ic_h_w_size, offset);
            output_values.y = asfloat( input.Load( offset*4 ) );

        } else if (remain_channel == 1) {
            int offset      = buffer_offset;
            output_values.x = asfloat( input.Load( offset*4 ) );
        }
    }

    output[DTid.xy] = output_values;

}