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
#define UP_DIV(A, B) (((A) + (B) - 1) / (B))

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

[numthreads(8, 8, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    uint output_cw_idx = DTid.x;
    uint output_bh_idx = DTid.y;

    uint round_h = UP_DIV(out_shape[2], 2);
    uint round_w = UP_DIV(out_shape[3], 2);
    uint batch_round_h = out_shape[0] * round_h;

    if (output_cw_idx >= UP_DIV(in_shape[1], 4)*round_w || output_bh_idx >= batch_round_h) {
        return;
    }

    const uint c_block_idx = output_cw_idx / round_w;
    const uint w_block_idx = output_cw_idx - round_w * c_block_idx;
    const uint batch = output_bh_idx / round_h;
    const uint h_block_idx = output_bh_idx - round_h * batch;

    const int width_start_idx = (w_block_idx << 1) - padding_wh.x;
    const int height_start_idx = (h_block_idx << 1) - padding_wh.y;

    int4 unit_offsets = {0,1,2,3};
    const int4 width_idx  = unit_offsets + width_start_idx;
    const int4 height_idx = unit_offsets + height_start_idx;

    int in_width = in_shape[2];
    int in_height = in_shape[3];
    int4 in_wc_idx = mad(c_block_idx, in_width, width_idx);
    int4 in_bh_idx = mad(batch, in_height, height_idx);

    in_wc_idx = width_idx >= (int4)(0) && width_idx < (int4)(in_width) ? in_wc_idx : (int4)(-1);
    in_bh_idx = height_idx >= (int4)(0) && height_idx < (int4)(in_height) ? in_bh_idx : (int4)(-1);

    float4 v_in[4][4];
    [unroll] for (int j = 0; j < 4; ++j) {
        [unroll] for (int i = 0; i < 4; ++i) {
            uint2 pos = {in_wc_idx[j], in_bh_idx[i]};
            v_in[j][i] = input[pos];
        }
    }

    float4 v_m[4][4];

    v_m[0][0] = v_in[0][0] - v_in[0][2];
    v_m[1][0] = v_in[1][0] - v_in[1][2];
    v_m[2][0] = v_in[2][0] - v_in[2][2];
    v_m[3][0] = v_in[3][0] - v_in[3][2];

    v_m[0][1] = 0.5f * v_in[0][1] + 0.5f * v_in[0][2];
    v_m[1][1] = 0.5f * v_in[1][1] + 0.5f * v_in[1][2];
    v_m[2][1] = 0.5f * v_in[2][1] + 0.5f * v_in[2][2];
    v_m[3][1] = 0.5f * v_in[3][1] + 0.5f * v_in[3][2];

    v_m[0][2] = -0.5f * v_in[0][1] + 0.5f * v_in[0][2];
    v_m[1][2] = -0.5f * v_in[1][1] + 0.5f * v_in[1][2];
    v_m[2][2] = -0.5f * v_in[2][1] + 0.5f * v_in[2][2];
    v_m[3][2] = -0.5f * v_in[3][1] + 0.5f * v_in[3][2];

    v_m[0][3] = -v_in[0][1] + v_in[0][3];
    v_m[1][3] = -v_in[1][1] + v_in[1][3];
    v_m[2][3] = -v_in[2][1] + v_in[2][3];
    v_m[3][3] = -v_in[3][1] + v_in[3][3];

    [unroll] for (int j = 0; j < 4; ++j) {
        uint2 pos0 = {output_cw_idx, output_bh_idx + (j * 4 + 0) * batch_round_h};
        matrix_v[pos0] = v_m[0][j] - v_m[2][j];
        uint2 pos1 = {output_cw_idx, output_bh_idx + (j * 4 + 1) * batch_round_h};
        matrix_v[pos1] = 0.5f * v_m[1][j] + 0.5f * v_m[2][j];
        uint2 pos2 = {output_cw_idx, output_bh_idx + (j * 4 + 2) * batch_round_h};
        matrix_v[pos2] = -0.5f * v_m[1][j] + 0.5f * v_m[2][j];
        uint2 pos3 = {output_cw_idx, output_bh_idx + (j * 4 + 3) * batch_round_h};
        matrix_v[pos3] = -v_m[1][j] + v_m[3][j];
    }
}
