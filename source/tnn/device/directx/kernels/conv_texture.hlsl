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

#define READ_INPUT_IMAGE(i, base) \
    int in_width_value##i = in_width##i + base; \
    in_width_value##i = \
           (in_width_value##i < 0 || in_width_value##i >= in_shape[3]) ? -1 : in_idx + in_width_value##i; \
    int2 pos_in##i = {in_width_value##i, in_hb_value}; \
    in##i = input[pos_in##i];

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

    vector<uint, 4> kernel_wh;

    vector<uint, 4> stride_wh;

    vector<uint, 4> padding_wh;

    vector<uint, 4> dilation_wh;

    vector<uint, 4> activation_type;

};


Texture2D<float4> input : register(t0);
Texture2D<float4> weights : register(t1);
Texture2D<float4> bias : register(t2);
RWTexture2D<float4> output : register(u0);

[numthreads(4, 4, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    int in_channel_block_length = UP_DIV(in_shape[1], 4);
    int out_width_blocks = UP_DIV(out_shape[3], 4);

    int output_cw_idx = DTid.x;
    int output_bh_idx = DTid.y;

    if (output_cw_idx > UP_DIV(out_shape[1], 4)*UP_DIV(out_shape[3], 4) || output_bh_idx > out_shape[0]*out_shape[2]) {
        return;
    }

    int out_channel_block_idx = output_cw_idx / out_width_blocks;
    int out_width_block_idx   = output_cw_idx % out_width_blocks;

    int2 pos_bias = {out_channel_block_idx, 0};
    float4 out0 = bias[pos_bias];
    float4 out1 = out0;
    float4 out2 = out0;
    float4 out3 = out0;

    int in_width0 = mad(out_width_block_idx, stride_wh[0] << 2, -padding_wh[0]);
    int in_width1 = in_width0 + stride_wh[0];
    int in_width2 = in_width0 + stride_wh[0] * 2;
    int in_width3 = in_width0 + stride_wh[0] * 3;

    int height_start = mad((output_bh_idx % out_shape[2]), stride_wh[1], -padding_wh[1]);
    int in_height_start = mad(height_start < 0 ? (-height_start + dilation_wh[1] - 1) / dilation_wh[1] : 0 , dilation_wh[1], height_start);
    int in_height_end = min(mad(kernel_wh[1], dilation_wh[1], height_start), in_shape[2]);

    int batch_idx = mul((output_bh_idx / out_shape[2]), in_shape[2]);
    int weights_h_idx = mul(out_channel_block_idx, mul(kernel_wh[0], kernel_wh[1])) +
                        mul(height_start < 0 ? (-height_start + dilation_wh[1] - 1) / dilation_wh[1] : 0 , kernel_wh[0]);

    float4 in0, in1, in2, in3;
    float4 weights0, weights1, weights2, weights3;

    for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
        int in_idx  = mul(input_c_block_idx, in_shape[3]);
        int weights_x_idx = input_c_block_idx << 2;
        int weights_y_idx = weights_h_idx;
        for (int iy = in_height_start; iy < in_height_end; iy += dilation_wh[1]) {
             int in_hb_value = iy + batch_idx;
             for (int w = 0; w < kernel_wh[0]; w++) {
                 int input_w_base = mul(w, dilation_wh[0]);

                 READ_INPUT_IMAGE(0, input_w_base);
                 READ_INPUT_IMAGE(1, input_w_base);
                 READ_INPUT_IMAGE(2, input_w_base);
                 READ_INPUT_IMAGE(3, input_w_base);

                 int2 pos0 = {weights_x_idx, weights_y_idx};
                 weights0 = weights[pos0];
                 int2 pos1 = {weights_x_idx + 1, weights_y_idx};
                 weights1 = weights[pos1];
                 int2 pos2 = {weights_x_idx + 2, weights_y_idx};
                 weights2 = weights[pos2];
                 int2 pos3 = {weights_x_idx + 3, weights_y_idx++};
                 weights3 = weights[pos3];

                 CALCULATE_OUTPUT(0);
                 CALCULATE_OUTPUT(1);
                 CALCULATE_OUTPUT(2);
                 CALCULATE_OUTPUT(3);
             }
        }
    }

    ActivationProcess(out0, activation_type[0]);
    ActivationProcess(out1, activation_type[0]);
    ActivationProcess(out2, activation_type[0]);
    ActivationProcess(out3, activation_type[0]);

    int out_x_base = mul(out_channel_block_idx, out_shape[3]);
    int out_x_idx  = out_width_block_idx << 2;

    int remain = out_shape[3] - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;

    if (remain >= 4) {
        int2 pos_out0 = {output_w_idx, output_bh_idx};
        output[pos_out0] = out0;
        int2 pos_out1 = {output_w_idx + 1, output_bh_idx};
        output[pos_out1] = out1;
        int2 pos_out2 = {output_w_idx + 2, output_bh_idx};
        output[pos_out2] = out2;
        int2 pos_out3 = {output_w_idx + 3, output_bh_idx};
        output[pos_out3] = out3;
    } else if (remain == 3) {
        int2 pos_out0 = {output_w_idx, output_bh_idx};
        output[pos_out0] = out0;
        int2 pos_out1 = {output_w_idx + 1, output_bh_idx};
        output[pos_out1] = out1;
        int2 pos_out2 = {output_w_idx + 2, output_bh_idx};
    } else if (remain == 2) {
        int2 pos_out0 = {output_w_idx, output_bh_idx};
        output[pos_out0] = out0;
        int2 pos_out1 = {output_w_idx + 1, output_bh_idx};
    } else if (remain == 1) {
        int2 pos_out0 = {output_w_idx, output_bh_idx};
        output[pos_out0] = out0;
    }

}