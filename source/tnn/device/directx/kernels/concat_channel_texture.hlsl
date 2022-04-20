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

#define UP_DIV(A, B) (((A) + (B) - 1) / B)

Texture2D<float4> input0 : register(t0);
Texture2D<float4> input1 : register(t1);
RWTexture2D<float4> output : register(u0);

cbuffer Shapes: register( b0 )
{
    // NB
    // in a constant buffer, each element of an array must start on a 4-float boundary.
    // so we choose float4 for the ease of alignment with cpp

    // input0 dimension
    vector<int, 4> id0;

    // output dimension
    vector<int, 4> od;

    vector<int, 4> param;
};

[numthreads(16, 16, 16)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    int input0_channel = id0[1];
    int output_channel = od[1];
    int channel0_mod_4 = param[0];

    int width_idx    = DTid.y;
    int channel_block_idx = DTid.z;
    int hb_idx       = DTid.x;

    if ( width_idx >= od[3] || channel_block_idx >= UP_DIV(od[1], 4) || hb_idx >= od[2]*od[0] ) {
        return;
    }

    int width = od[3];
    int input1_channel = output_channel - input0_channel;
    int input0_channel_blk = (input0_channel + 3) >> 2;

    float4 data = {0, 0, 0, 0};
    if (channel_block_idx < input0_channel_blk - 1) {
        int2 pos_in0 = {mad(channel_block_idx, width, width_idx), hb_idx};
        data = input0[pos_in0];

    } else if(channel_block_idx == input0_channel_blk - 1) {
        int2 pos_in0 = {mad(channel_block_idx, width, width_idx), hb_idx};
        float4 data0 = input0[pos_in0];

        int2 pos_in1 = {width_idx, hb_idx};
        float4 data1 = input1[pos_in1];

        if (channel0_mod_4 == 1){
            data = float4(data0.x, data1.x, data1.y, data1.z);
        } else if (channel0_mod_4 == 2){
            data = float4(data0.x, data0.y, data1.x, data1.y);
        } else {
            data = float4(data0.x, data0.y, data0.z, data1.x);
        }

    } else {
        int input1_channel_idx = channel_block_idx - input0_channel_blk;

        int2 pos_in0 = {mad(input1_channel_idx, width, width_idx), hb_idx};
        float4 data0 = input1[pos_in0];

        float4 data1 = {0, 0, 0, 0};
        if (((input1_channel_idx + 1) << 2) < input1_channel) {
            int2 pos_in1 = {mad((input1_channel_idx + 1), width, width_idx), hb_idx};
            float4 data1 = input1[pos_in1];
        }

        if (channel0_mod_4 == 1){
            data = float4(data0.w, data1.x, data1.y, data1.z);
        } else if (channel0_mod_4 == 2){
            data = float4(data0.z, data0.w, data1.x, data1.y);
        } else {
            data = float4(data0.y, data0.z, data0.w, data1.x);
        }
    }

    int2 pos_out = {mad(channel_block_idx, width, width_idx), hb_idx};
    output[pos_out] = data;
}
