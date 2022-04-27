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

#define INT_MIN -2147483648

#define CALCULATE_OUTPUT(i)                  \
    out##i = mad(in##i.x, weights0, out##i); \
    out##i = mad(in##i.y, weights1, out##i); \
    out##i = mad(in##i.z, weights2, out##i); \
    out##i = mad(in##i.w, weights3, out##i);

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

    vector<uint, 4> stride_wh;

    vector<uint, 4> activation_type;

};

Texture2D<float4> input : register(t0);
Texture2D<float4> weights : register(t1);
Texture2D<float4> bias : register(t2);
RWTexture2D<float4> output : register(u0);

[numthreads(4, 4, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    int2 wh = {in_shape[3], in_shape[2]};
    int input_c_blocks = UP_DIV(in_shape[1] ,4);
    int output_w_updiv_4 = UP_DIV(out_shape[3] ,4);

    int output_cw_idx = DTid.x;
    int bh_idx = DTid.y;

    if (output_cw_idx >= UP_DIV(out_shape[1], 4)*UP_DIV(out_shape[3], 4) || bh_idx >= out_shape[0]*out_shape[2]) {
        return;
    }

    int output_c_block_idx = output_cw_idx / output_w_updiv_4;
    int output_w_block_idx = output_cw_idx % output_w_updiv_4;

    int2 pos_bias = {output_c_block_idx, 0};
    float4 out0 = bias[pos_bias];
    float4 out1 = out0;
    float4 out2 = out0;
    float4 out3 = out0;

    int input_w_idx0 = output_w_block_idx << 2;
    int input_w_idx1 = input_w_idx0 + 1;
    int input_w_idx2 = input_w_idx0 + 2;
    int input_w_idx3 = input_w_idx0 + 3;

    input_w_idx0 = input_w_idx0 >= wh.x ? INT_MIN : input_w_idx0;
    input_w_idx1 = input_w_idx1 >= wh.x ? INT_MIN : input_w_idx1;
    input_w_idx2 = input_w_idx2 >= wh.x ? INT_MIN : input_w_idx2;
    input_w_idx3 = input_w_idx3 >= wh.x ? INT_MIN : input_w_idx3;

    float4 in0, in1, in2, in3;
    float4 weights0, weights1, weights2, weights3;
    int input_w_base   = 0;
    int weights_w_base = 0;

    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {

        int2 pos_in0 = {input_w_base + input_w_idx0, bh_idx};
        in0 = input[pos_in0];
        int2 pos_in1 = {input_w_base + input_w_idx1, bh_idx};
        in1 = input[pos_in1];
        int2 pos_in2 = {input_w_base + input_w_idx2, bh_idx};
        in2 = input[pos_in2];
        int2 pos_in3 = {input_w_base + input_w_idx3, bh_idx};
        in3 = input[pos_in3];

        int2 pos_w0 = {weights_w_base, output_c_block_idx};
        weights0 = weights[pos_w0];
        int2 pos_w1 = {weights_w_base + 1, output_c_block_idx};
        weights1 = weights[pos_w1];
        int2 pos_w2 = {weights_w_base + 2, output_c_block_idx};
        weights2 = weights[pos_w2];
        int2 pos_w3 = {weights_w_base + 3, output_c_block_idx};
        weights3 = weights[pos_w3];

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);

        input_w_base   += wh.x;
        weights_w_base += 4;
    }

    ActivationProcess(out0, activation_type[0]);
    ActivationProcess(out1, activation_type[0]);
    ActivationProcess(out2, activation_type[0]);
    ActivationProcess(out3, activation_type[0]);

    int out_x_base = mul(output_c_block_idx, wh.x);
    int out_x_idx = output_w_block_idx << 2;

    const int remain = wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;

    if (remain >= 4) {
        int2 pos_out0 = {output_w_idx, bh_idx};
        output[pos_out0] = out0;
        int2 pos_out1 = {output_w_idx + 1, bh_idx};
        output[pos_out1] = out1;
        int2 pos_out2 = {output_w_idx + 2, bh_idx};
        output[pos_out2] = out2;
        int2 pos_out3 = {output_w_idx + 3, bh_idx};
        output[pos_out3] = out3;
    } else if (remain == 3) {
        int2 pos_out0 = {output_w_idx, bh_idx};
        output[pos_out0] = out0;
        int2 pos_out1 = {output_w_idx + 1, bh_idx};
        output[pos_out1] = out1;
        int2 pos_out2 = {output_w_idx + 2, bh_idx};
        output[pos_out2] = out2;
    } else if (remain == 2) {
        int2 pos_out0 = {output_w_idx, bh_idx};
        output[pos_out0] = out0;
        int2 pos_out1 = {output_w_idx + 1, bh_idx};
        output[pos_out1] = out1;
    } else if (remain == 1) {
        int2 pos_out0 = {output_w_idx, bh_idx};
        output[pos_out0] = out0;
    }

}