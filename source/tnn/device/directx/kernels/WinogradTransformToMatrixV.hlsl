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


Texture2D<float4> input : register(t0);
RWTexture2D<float4> matrix_v : register(u0);

[numthreads(4, 4, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    int in_height = in_shape[2];
    int in_width = in_shape[3];
    int in_channel = in_shape[1];
    int round_h = UP_DIV(out_shape[2], 2);
    int round_w = UP_DIV(out_shape[3], 2);
    int global_size_dim1 = out_shape[0]*round_h;

    int output_cw_idx = DTid.x; //c/4 w/2
    int output_bh_idx = DTid.y; //b h/2

    if (output_cw_idx >= UP_DIV(in_channel, 4)*round_w || output_bh_idx >= out_shape[0]*round_h) {
        return;
    }

    int c_block_idx = output_cw_idx / round_w;
    int w_block_idx = output_cw_idx - mul(c_block_idx, round_w);
    int batch = output_bh_idx / round_h;
    int h_block_idx = output_bh_idx - mul(batch, round_h);

    int width_start_idx = (w_block_idx << 1) - padding_wh[0];
    int height_start_idx = (h_block_idx << 1) - padding_wh[1];

    int4 width_idx = (int4)(width_start_idx) + (int4)(0,1,2,3);
    int4 height_idx  = (int4)(height_start_idx) + (int4)(0,1,2,3);

    int4 in_wc_idx = mad((int4)(c_block_idx), (int4)(in_width), width_idx);
    int4 in_bh_idx = mad((int4)(batch), (int4)(in_height), height_idx);

    in_wc_idx = width_idx < (int4)(0) || width_idx >= (int4)(in_width) ? (int4)(-1) : in_wc_idx;
    in_bh_idx = height_idx < (int4)(0) || height_idx >= (int4)(in_height) ? (int4)(-1) : in_bh_idx;

    int2 pos_in00 = {in_wc_idx.x, in_bh_idx.x};
    float4 in00 = input[pos_in00];
    int2 pos_in10 = {in_wc_idx.y, in_bh_idx.x};
    float4 in10 = input[pos_in10];
    int2 pos_in20 = {in_wc_idx.z, in_bh_idx.x};
    float4 in20 = input[pos_in20];
    int2 pos_in30 = {in_wc_idx.w, in_bh_idx.x};
    float4 in30 = input[pos_in30];

    int2 pos_in01 = {in_wc_idx.x, in_bh_idx.y};
    float4 in01 = input[pos_in01];
    int2 pos_in11 = {in_wc_idx.y, in_bh_idx.y};
    float4 in11 = input[pos_in11];
    int2 pos_in21 = {in_wc_idx.z, in_bh_idx.y};
    float4 in21 = input[pos_in21];
    int2 pos_in31 = {in_wc_idx.w, in_bh_idx.y};
    float4 in31 = input[pos_in31];

    int2 pos_in02 = {in_wc_idx.x, in_bh_idx.z};
    float4 in02 = input[pos_in02];
    int2 pos_in12 = {in_wc_idx.y, in_bh_idx.z};
    float4 in12 = input[pos_in12];
    int2 pos_in22 = {in_wc_idx.z, in_bh_idx.z};
    float4 in22 = input[pos_in22];
    int2 pos_in32 = {in_wc_idx.w, in_bh_idx.z};
    float4 in32 = input[pos_in32];

    int2 pos_in03 = {in_wc_idx.x, in_bh_idx.w};
    float4 in03 = input[pos_in03];
    int2 pos_in13 = {in_wc_idx.y, in_bh_idx.w};
    float4 in13 = input[pos_in13];
    int2 pos_in23 = {in_wc_idx.z, in_bh_idx.w};
    float4 in23 = input[pos_in23];
    int2 pos_in33 = {in_wc_idx.w, in_bh_idx.w};
    float4 in33 = input[pos_in33];

    float4 v00 = in00 - in02;
    float4 v10 = in10 - in12;
    float4 v20 = in20 - in22;
    float4 v30 = in30 - in32;

    float4 v01 = 0.5 * in01 + 0.5 * in02;
    float4 v11 = 0.5 * in11 + 0.5 * in12;
    float4 v21 = 0.5 * in21 + 0.5 * in22;
    float4 v31 = 0.5 * in31 + 0.5 * in32;

    float4 v02 = -0.5 * in01 + 0.5 * in02;
    float4 v12 = -0.5 * in11 + 0.5 * in12;
    float4 v22 = -0.5 * in21 + 0.5 * in22;
    float4 v32 = -0.5 * in31 + 0.5 * in32;

    float4 v03 = -in01 + in03;
    float4 v13 = -in11 + in13;
    float4 v23 = -in21 + in23;
    float4 v33 = -in31 + in33;

    int2 pos_out00 = {output_cw_idx, output_bh_idx};
    matrix_v[pos_out00] = v00 - v20;
    int2 pos_out10 = {output_cw_idx, mad(1, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out10] = 0.5 * v10 + 0.5 * v20;
    int2 pos_out20 = {output_cw_idx, mad(2, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out20] = -0.5 * v10 + 0.5 * v20;
    int2 pos_out30 = {output_cw_idx, mad(3, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out30] = -v10 + v30;

    int2 pos_out01 = {output_cw_idx, mad(4, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out01] = v01 - v21;
    int2 pos_out11 = {output_cw_idx, mad(5, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out11] = 0.5 * v11 + 0.5 * v21;
    int2 pos_out21 = {output_cw_idx, mad(6, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out21] = -0.5 * v11 + 0.5 * v21;
    int2 pos_out31 = {output_cw_idx, mad(7, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out31] = -v11 + v31;

    int2 pos_out02 = {output_cw_idx, mad(8, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out02] = v02 - v22;
    int2 pos_out12 = {output_cw_idx, mad(9, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out12] = 0.5 * v12 + 0.5 * v22;
    int2 pos_out22 = {output_cw_idx, mad(10, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out22] = -0.5 * v12 + 0.5 * v22;
    int2 pos_out32 = {output_cw_idx, mad(11, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out32] = -v12 + v32;

    int2 pos_out03 = {output_cw_idx, mad(12, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out03] = v03 - v23;
    int2 pos_out13 = {output_cw_idx, mad(13, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out13] = 0.5 * v13 + 0.5 * v23;
    int2 pos_out23 = {output_cw_idx, mad(14, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out23] = -0.5 * v13 + 0.5 * v23;
    int2 pos_out33 = {output_cw_idx, mad(15, global_size_dim1, output_bh_idx)};
    matrix_v[pos_out33] = -v13 + v33;

}