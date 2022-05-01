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

#define ActivationType_None 0x0000
#define ActivationType_ReLU 0x0001
#define ActivationType_ReLU6 0x0002
#define ActivationType_SIGMOID_MUL 0x0100

#define ActivationProcess(output, activation_type) \
    output = (activation_type == ActivationType_ReLU) ? max(output, (float4)0) : \
    ((activation_type == ActivationType_ReLU6) ? clamp(output,(float4)0,(float4)6) : \
    ((activation_type == ActivationType_SIGMOID_MUL) ? rcp((float4)1 + mul(exp(-output), output)) : \
    output ))

cbuffer Shapes: register( b0 )
{
    // NB
    // in a constant buffer, each element of an array must start on a 4-float boundary.
    // so we choose float4 for the ease of alignment with cpp

    // input a dimension
    // in_shape[0] in_shape[1] in_shape[2] in_shape[3]
    //    in         ic          ih           iw
    //    x          y           z            w
    vector<uint, 4> in_shape;
    // output a dimension
    // out_shape[0] out_shape[1] out_shape[2] out_shape[3]
    //     on           oc           oh           ow
    //     x            y            z            w
    vector<uint, 4> out_shape;

    vector<uint, 4> kernel_wh;

    vector<uint, 4> stride_wh;

    vector<uint, 4> padding_wh;

    vector<uint, 4> dilation_wh;

    vector<uint, 4> activation_type;

};


Texture2D<float4> matrix_v : register(t0);
Texture2D<float4> matrix_u : register(t1);
RWTexture2D<float4> matrix_m : register(u0);

[numthreads(4, 4, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    int output_cw_block_idx = DTid.x; //c/4  w/2/4
    int output_16_bh_idx  = DTid.y; //16 b h/2

    int batch = out_shape[0];
    int round_h = UP_DIV(out_shape[2], 2);
    int round_w = UP_DIV(out_shape[3], 2);
    int round_4x4_w = UP_DIV(round_w, 4);
    int batch_round_h = batch * round_h;
    int out_channel_block = UP_DIV(out_shape[1], 4);
    int in_channel_block = UP_DIV(in_shape[1], 4);

    if (output_cw_block_idx >= out_channel_block*round_4x4_w || output_16_bh_idx >= 16*batch_round_h) {
        return;
    }

    int c_block_idx = output_cw_block_idx / round_4x4_w;
    int w_block_idx = output_cw_block_idx - mul(c_block_idx, round_4x4_w);
    int4 w_idx = (int4)(w_block_idx << 2) + (int4)(0,1,2,3);

    int alpha = output_16_bh_idx / batch_round_h;
    int u_bh_idx = mul(alpha, out_channel_block) + c_block_idx;

    float4 m0 = (float4)(0);
    float4 m1 = (float4)(0);
    float4 m2 = (float4)(0);
    float4 m3 = (float4)(0);

    for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block; ++input_c_block_idx) {

        int4 input_c_idx = (int4)(input_c_block_idx << 2) + (int4)(0,1,2,3);
        int4 v_cw_idx = w_idx >= (int4)(round_w) ? (int4)(-1) : mad((int4)(input_c_block_idx), (int4)(round_w), w_idx);

        int2 pos_v0 = {v_cw_idx.x, output_16_bh_idx};
        float4 v_in0 = matrix_v[pos_v0];
        int2 pos_v1 = {v_cw_idx.y, output_16_bh_idx};
        float4 v_in1 = matrix_v[pos_v1];
        int2 pos_v2 = {v_cw_idx.z, output_16_bh_idx};
        float4 v_in2 = matrix_v[pos_v2];
        int2 pos_v3 = {v_cw_idx.w, output_16_bh_idx};
        float4 v_in3 = matrix_v[pos_v3];

        int2 pos_u0 = {input_c_idx.x, u_bh_idx};
        float4 u_in0 = matrix_u[pos_u0];
        int2 pos_u1 = {input_c_idx.y, u_bh_idx};
        float4 u_in1 = matrix_u[pos_u1];
        int2 pos_u2 = {input_c_idx.z, u_bh_idx};
        float4 u_in2 = matrix_u[pos_u2];
        int2 pos_u3 = {input_c_idx.w, u_bh_idx};
        float4 u_in3 = matrix_u[pos_u3];

        m0 = mad(v_in0.x, u_in0, m0);
        m0 = mad(v_in0.y, u_in1, m0);
        m0 = mad(v_in0.z, u_in2, m0);
        m0 = mad(v_in0.w, u_in3, m0);

        m1 = mad(v_in1.x, u_in0, m1);
        m1 = mad(v_in1.y, u_in1, m1);
        m1 = mad(v_in1.z, u_in2, m1);
        m1 = mad(v_in1.w, u_in3, m1);

        m2 = mad(v_in2.x, u_in0, m2);
        m2 = mad(v_in2.y, u_in1, m2);
        m2 = mad(v_in2.z, u_in2, m2);
        m2 = mad(v_in2.w, u_in3, m2);

        m3 = mad(v_in3.x, u_in0, m3);
        m3 = mad(v_in3.y, u_in1, m3);
        m3 = mad(v_in3.z, u_in2, m3);
        m3 = mad(v_in3.w, u_in3, m3);
    }

    int output_cw_idx = mad(c_block_idx, round_w, w_idx.x);
    int2 pos_m0 = {output_cw_idx, output_16_bh_idx};
    matrix_m[pos_m0] = m0;

    if(w_idx.y < round_w) {
        int2 pos_m1 = {output_cw_idx+1, output_16_bh_idx};
        matrix_m[pos_m1] = m1;
    }

    if(w_idx.z < round_w) {
        int2 pos_m2 = {output_cw_idx+2, output_16_bh_idx};
        matrix_m[pos_m2] = m2;
    }

    if(w_idx.w < round_w) {
        int2 pos_m3 = {output_cw_idx+3, output_16_bh_idx};
        matrix_m[pos_m3] = m3;
    }

}