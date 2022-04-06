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

Texture2D<float4> input : register(t0);
RWTexture2D<float4> output : register(u0);

cbuffer Shapes: register( b0 )
{
    // NB
    // in a constant buffer, each element of an array must start on a 4-float boundary.
    // so we choose float4 for the ease of alignment with cpp

    // input dimension
    vector<int, 4> id;

    // output dimension
    vector<int, 4> od;

};

[numthreads(1, 1, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    int channel_block_idx = DTid.z;
    int w                 = DTid.y;
    int hb                = DTid.x;

    if (channel_block_idx >= od[1]  || w >= od[3] || hb >= od[0]*od[2]) {
        return;
    }

    int width = od[3];

    int pos_x =  mad(channel_block_idx, width, w);
    int2 pos = {pos_x,hb};
    float4 unary_in = input[pos];
    float4 unary_out = UNARY_OP;
    output[pos] = unary_out;

}

