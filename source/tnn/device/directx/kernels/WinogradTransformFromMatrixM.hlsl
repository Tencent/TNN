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

#define ActivationType_None 0x0000
#define ActivationType_ReLU 0x0001
#define ActivationType_ReLU6 0x0002
#define ActivationType_SIGMOID_MUL 0x0100

#define ActivationProcess(output, activation_type) \
    output = (activation_type == ActivationType_ReLU) ? max(output, (float4)0) : \
    ((activation_type == ActivationType_ReLU6) ? clamp(output,(float4)0,(float4)6) : \
    ((activation_type == ActivationType_SIGMOID_MUL) ? rcp((float4)1 + mul(exp(-output), output)) : \
    output ))

Texture2D<float4> matrix_m : register(t0);
Texture2D<float4> bias : register(t1);
RWTexture2D<float4> output : register(u0);

[numthreads(8, 8, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    uint output_cw_idx = DTid.x;
    uint output_bh_idx = DTid.y;

    uint round_h = UP_DIV(out_shape[2], 2);
    uint round_w = UP_DIV(out_shape[3], 2);
    uint batch_round_h = out_shape[0] * round_h;
    uint out_width = out_shape[3];
    uint out_height = out_shape[2];

    if (output_cw_idx >= UP_DIV(out_shape[1], 4)*round_w || output_bh_idx >= batch_round_h) {
        return;
    }

    const uint c_block_idx = output_cw_idx / round_w;
    const uint w_block_idx = output_cw_idx - round_w * c_block_idx;
    const uint batch = output_bh_idx / round_h;
    const uint h_block_idx = output_bh_idx - round_h * batch;

    uint2 bias_pos = {c_block_idx, 0};
    float4 bias_value = bias[bias_pos];

    float4 v_m[4][4];
    [unroll] for (int j = 0; j < 4; ++j) {
        [unroll] for (int i = 0; i < 4; ++i) {
            uint2 pos = {output_cw_idx, output_bh_idx + (j * 4 + i) * batch_round_h};
            v_m[i][j] = matrix_m[pos];
        }
    }

    float4 v_out[4][2];
    [unroll] for (int j = 0; j < 4; ++j) {
        v_out[j][0] = v_m[j][0] + v_m[j][1] + v_m[j][2];
        v_out[j][1] = v_m[j][1] - v_m[j][2] + v_m[j][3];
    }

    uint2 unit_offsets = {0,1};
    uint2 ow = (w_block_idx << 1) + unit_offsets;
    uint2 oh = (h_block_idx << 1) + unit_offsets;
    uint2 ox = mad(c_block_idx, out_width, ow);
    uint2 oy = mad(batch, out_height, oh);

    float4 res[2][2];
    [unroll] for (int j = 0; j < 2; ++j) {
        res[0][j] = bias_value + v_out[0][j] + v_out[1][j] + v_out[2][j];
        res[1][j] = bias_value + v_out[1][j] - v_out[2][j] + v_out[3][j];
        res[0][j] = ActivationProcess(res[0][j], activation_type[0]);
        res[1][j] = ActivationProcess(res[1][j], activation_type[0]);
    }

    [unroll] for (int j = 0; j < 2; ++j) {
        [unroll] for (int i = 0; i < 2; ++i) {
            if (ow[j] < out_width && oh[i] < out_height) {
                uint2 pos_out = {ox[j], oy[i]};
                output[pos_out] = res[j][i];
            }
        }
    }
}
