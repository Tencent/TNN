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

cbuffer Shapes: register( b0 )
{
    // NB
    // in a constant buffer, each element of an array must start on a 4-float boundary.
    // so we choose float4 for the ease of alignment with cpp

    // input a dimension
    // ad_0[0] ad_0[1] ad_0[2] ad_3[0]
    //   i0n     i0c     i0h     i0w
    //    x       y       z       w
    vector<uint, 4> ad_0;
    vector<uint, 4> ad_3;

    // input b dimension
    // bd_0[0] bd_0[1] bd_0[2] bd_3[0]
    //   i1n     i1c     i1h     i1w
    //    x       y       z       w
    vector<uint, 4> bd_0;
    vector<uint, 4> bd_3;

    // output dimension
    // od_0[0] od_0[1] od_0[2] od_3[0]
    //   on      oc      oh      ow
    //    x       y       z       w
    vector<uint, 4> od_0;
    vector<uint, 4> od_3;
};


Texture2D<float4> input0 : register(t0);
Texture2D<float4> input1 : register(t1);
RWTexture2D<float4> output : register(u0);

[numthreads(1, 1, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    uint output_cw = DTid.x;
    uint output_bh = DTid.y;

    if (output_cw >= od_0[1]*od_3[0] || output_bh >= od_0[0]*od_0[2]) {
        return;
    }

    uint output_h_idx = output_bh % od_0[2];
    uint output_b_idx = output_bh / od_0[2];
    uint output_w_idx = output_cw % od_3[0];
    uint output_c_4_idx = output_cw / od_3[0];
    uint input0_c_4_blocks = UP_DIV(ad_0[1], 4);
    uint input1_c_4_blocks = UP_DIV(bd_0[1], 4);

    float4 in0, in1;
    uint input0_h_idx, input0_b_idx, input0_w_idx, input0_c_4_idx, input0_c_idx0;
    input0_h_idx = ad_0[2] == 1 ? 0 : output_h_idx;
    input0_b_idx = ad_0[0] == 1 ? 0 : output_b_idx;
    input0_w_idx = ad_3[0] == 1 ? 0 : output_w_idx;
    input0_c_4_idx = output_c_4_idx < input0_c_4_blocks ? output_c_4_idx : input0_c_4_blocks - 1;
    input0_c_idx0 = input0_c_4_idx << 2;
    uint2 pos_i0 = {input0_c_4_idx * ad_3[0] + input0_w_idx, input0_b_idx * ad_0[2] + input0_h_idx};
    in0 = input0[pos_i0];
    if (ad_0[1] == 1) {
        in0.y = in0.x;
        in0.z = in0.x;
        in0.w = in0.x;
    }

    uint input1_h_idx, input1_b_idx, input1_w_idx, input1_c_4_idx, input1_c_idx0;
    input1_h_idx = bd_0[2] == 1 ? 0 : output_h_idx;
    input1_b_idx = bd_0[0] == 1 ? 0 : output_b_idx;
    input1_w_idx = bd_3[0] == 1 ? 0 : output_w_idx;
    input1_c_4_idx = output_c_4_idx < input1_c_4_blocks ? output_c_4_idx : input1_c_4_blocks - 1;
    input1_c_idx0 = input1_c_4_idx << 2;
    uint2 pos_i1 = {input1_c_4_idx * bd_3[0] + input1_w_idx, input1_b_idx * bd_0[2] + input1_h_idx};
    in1 = input1[pos_i1];
    if (bd_0[1] == 1) {
        in1.y = in1.x;
        in1.z = in1.x;
        in1.w = in1.x;
    }

    output[DTid.xy] = in0 BINARY_OP in1;
}