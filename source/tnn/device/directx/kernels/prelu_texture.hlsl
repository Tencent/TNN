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

Texture2D<float4> input : register(t0);
Texture2D<float4> scope : register(t1);
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
    int width = od[3];
    int cw_idx      = DTid.x;
    int bh_idx      = DTid.y;
    int c_block_idx = cw_idx / width;

    if (cw_idx >= UP_DIV(od[1], 4)*od[3] || bh_idx >= od[0]*od[2]) {
        return;
    }

    int2 pos_in = {cw_idx, bh_idx};
    float4 in_data = input[pos_in];
    int2 pos_scope = {c_block_idx, 0};
    float4 scope_data = scope[pos_scope];

    float4 out_data = in_data < 0 ? in_data * scope_data : in_data;

    int2 pos_out = {cw_idx, bh_idx};
    output[pos_out] = out_data;

}