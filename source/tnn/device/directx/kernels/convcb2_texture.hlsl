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

#define CALCULATE_SLICE_OUTPUT(s_idx) \
    out_w0_s##s_idx += mul(weights_c0_s##s_idx, in0.x); \
    out_w1_s##s_idx += mul(weights_c0_s##s_idx, in1.x); \
    out_w2_s##s_idx += mul(weights_c0_s##s_idx, in2.x); \
    out_w3_s##s_idx += mul(weights_c0_s##s_idx, in3.x); \
    out_w0_s##s_idx += mul(weights_c1_s##s_idx, in0.y); \
    out_w1_s##s_idx += mul(weights_c1_s##s_idx, in1.y); \
    out_w2_s##s_idx += mul(weights_c1_s##s_idx, in2.y); \
    out_w3_s##s_idx += mul(weights_c1_s##s_idx, in3.y); \
    out_w0_s##s_idx += mul(weights_c2_s##s_idx, in0.z); \
    out_w1_s##s_idx += mul(weights_c2_s##s_idx, in1.z); \
    out_w2_s##s_idx += mul(weights_c2_s##s_idx, in2.z); \
    out_w3_s##s_idx += mul(weights_c2_s##s_idx, in3.z); \
    out_w0_s##s_idx += mul(weights_c3_s##s_idx, in0.w); \
    out_w1_s##s_idx += mul(weights_c3_s##s_idx, in1.w); \
    out_w2_s##s_idx += mul(weights_c3_s##s_idx, in2.w); \
    out_w3_s##s_idx += mul(weights_c3_s##s_idx, in3.w); \

#define WriteSliceOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx, output_h_idx, remain, idx) \
    int2 pos_out_0_##idx = {output_w_idx, output_h_idx}; \
    output[pos_out_0_##idx] = out0; \
    if (remain >= 2) { \
        int2 pos_out_1_##idx = {output_w_idx + 1, output_h_idx}; \
        output[pos_out_1_##idx] = out1; \
    } \
    if (remain >= 3) { \
        int2 pos_out_2_##idx = {output_w_idx + 2, output_h_idx}; \
        output[pos_out_2_##idx] = out2; \
    } \
    if (remain >= 4) { \
        int2 pos_out_3_##idx = {output_w_idx + 3, output_h_idx}; \
        output[pos_out_3_##idx] = out3; \
    }

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

[numthreads(16, 16, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    int in_channel_block_length = UP_DIV(in_shape[1], 4);
    int out_channel_block_length = UP_DIV(out_shape[1], 4);
    int kernel_size = kernel_wh[0]*kernel_wh[1];
    int out_width_blocks = UP_DIV(out_shape[3], 4);

    int2 input_wh = {in_shape[3], in_shape[2]};
    int2 output_wh = {out_shape[3], out_shape[2]};

    int output_channel_slice_w_idx = DTid.x;
    int output_bh_idx = DTid.y;

    if (output_channel_slice_w_idx > UP_DIV(out_shape[1], 8)*UP_DIV(out_shape[3], 4) || output_bh_idx > out_shape[0]*out_shape[2]) {
         return;
    }

    int out_channel_slice_idx = output_channel_slice_w_idx / out_width_blocks;
    int out_channel_block_idx = out_channel_slice_idx << 1;
    int out_width_block_idx   = output_channel_slice_w_idx % out_width_blocks;

    int2 pos_bias0 = {out_channel_block_idx, 0};
    float4 out_w0_s0 = bias[pos_bias0];
    float4 out_w1_s0 = out_w0_s0;
    float4 out_w2_s0 = out_w0_s0;
    float4 out_w3_s0 = out_w0_s0;

    int2 pos_bias1 = {out_channel_block_idx + 1, 0};
    float4 out_w0_s1 = bias[pos_bias1];
    float4 out_w1_s1 = out_w0_s1;
    float4 out_w2_s1 = out_w0_s1;
    float4 out_w3_s1 = out_w0_s1;

    int in_width0 = mad(out_width_block_idx, stride_wh[0] << 2, -padding_wh[0]);
    int in_width1 = in_width0 + stride_wh[0];
    int in_width2 = in_width1 + stride_wh[0];
    int in_width3 = in_width2 + stride_wh[0];

    int height_start = mad((output_bh_idx % output_wh.y), stride_wh[1], -padding_wh[1]);
    int in_height_start = mad(height_start < 0 ? ((0-height_start + dilation_wh[1] - 1) / dilation_wh[1]) : 0 , dilation_wh[1], height_start);
    int in_height_end = min(mad(kernel_wh[1], dilation_wh[1], height_start), input_wh.y);

    int batch_idx = mul((output_bh_idx / output_wh.y), input_wh.y);
    int weights_y_idx_s0 = mad(out_channel_block_idx, kernel_size,
                               mul(height_start < 0 ? ((0-height_start + dilation_wh[1] - 1) / dilation_wh[1]) : 0 , kernel_wh[0]));
    int weights_y_idx_s1 = weights_y_idx_s0 + kernel_size;
    int2 weights_y_idx = {weights_y_idx_s0, weights_y_idx_s1};

    float4 in0, in1, in2, in3;
    float4 weights_c0_s0, weights_c1_s0, weights_c2_s0, weights_c3_s0;
    float4 weights_c0_s1, weights_c1_s1, weights_c2_s1, weights_c3_s1;
    for (int iy = in_height_start; iy < in_height_end; iy += dilation_wh[1]) {
        int in_hb_value = iy + batch_idx;
        int4 in_width = {in_width0, in_width1, in_width2, in_width3};
        for (int w = 0; w < kernel_wh[0]; w++) {
            int4 weights_x_idx = {0, 1, 2, 3};
            for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
                int in_idx  = mul(input_c_block_idx, input_wh.x);
                int4 is_w_in_boundary = (in_width >= 0 && in_width < input_wh.x);

                int4 in_cw_value = in_width + in_idx;

                int2 pos_in0 = {is_w_in_boundary.x ? in_cw_value.x : -1, in_hb_value};
                in0 = input[pos_in0];
                int2 pos_in1 = {is_w_in_boundary.y ? in_cw_value.y : -1, in_hb_value};
                in1 = input[pos_in1];
                int2 pos_in2 = {is_w_in_boundary.z ? in_cw_value.z : -1, in_hb_value};
                in2 = input[pos_in2];
                int2 pos_in3 = {is_w_in_boundary.w ? in_cw_value.w : -1, in_hb_value};
                in3 = input[pos_in3];

                int2 pos_w_c0_s0 = {weights_x_idx.x, weights_y_idx.x};
                weights_c0_s0 = weights[pos_w_c0_s0];
                int2 pos_w_c1_s0 = {weights_x_idx.y, weights_y_idx.x};
                weights_c1_s0 = weights[pos_w_c1_s0];
                int2 pos_w_c2_s0 = {weights_x_idx.z, weights_y_idx.x};
                weights_c2_s0 = weights[pos_w_c2_s0];
                int2 pos_w_c3_s0 = {weights_x_idx.w, weights_y_idx.x};
                weights_c3_s0 = weights[pos_w_c3_s0];

                int2 pos_w_c0_s1 = {weights_x_idx.x, weights_y_idx.y};
                weights_c0_s1 = weights[pos_w_c0_s1];
                int2 pos_w_c1_s1 = {weights_x_idx.y, weights_y_idx.y};
                weights_c1_s1 = weights[pos_w_c1_s1];
                int2 pos_w_c2_s1 = {weights_x_idx.z, weights_y_idx.y};
                weights_c2_s1 = weights[pos_w_c2_s1];
                int2 pos_w_c3_s1 = {weights_x_idx.w, weights_y_idx.y};
                weights_c3_s1 = weights[pos_w_c3_s1];

                CALCULATE_SLICE_OUTPUT(0);
                CALCULATE_SLICE_OUTPUT(1);

                weights_x_idx += 4;
            }

            weights_y_idx++;
            in_width += dilation_wh[0];
        }
    }

    ActivationProcess(out_w0_s0, activation_type[0]);
    ActivationProcess(out_w1_s0, activation_type[0]);
    ActivationProcess(out_w2_s0, activation_type[0]);
    ActivationProcess(out_w3_s0, activation_type[0]);

    ActivationProcess(out_w0_s1, activation_type[0]);
    ActivationProcess(out_w1_s1, activation_type[0]);
    ActivationProcess(out_w2_s1, activation_type[0]);
    ActivationProcess(out_w3_s1, activation_type[0]);

    int out_x_base = mul(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    int remain = output_wh.x - out_x_idx;
    int output_w_idx_s0 = out_x_base + out_x_idx;
//     WriteSliceOutputAntiOutOfBounds(output, out_w0_s0, out_w1_s0,
//                                     out_w2_s0, out_w3_s0, output_w_idx_s0,
//                                     output_bh_idx, remain, 1);

    int2 pos_out0_0 = {output_w_idx_s0, output_bh_idx};
    output[pos_out0_0] = out_w0_s0;
    if (remain >= 2) {
        int2 pos_out0_1 = {output_w_idx_s0 + 1, output_bh_idx};
        output[pos_out0_1] = out_w1_s0;
    }
    if (remain >= 3) {
        int2 pos_out0_2 = {output_w_idx_s0 + 2, output_bh_idx};
        output[pos_out0_2] = out_w2_s0;
    }
    if (remain >= 4) {
        int2 pos_out0_3 = {output_w_idx_s0 + 3, output_bh_idx};
        output[pos_out0_3] = out_w3_s0;
    }


    bool is_s1_in_boundary = (out_channel_block_idx + 1 < out_channel_block_length);
    if (!is_s1_in_boundary) return;
    int output_w_idx_s1 = output_w_idx_s0 + output_wh.x;
//     WriteSliceOutputAntiOutOfBounds(output, out_w0_s1, out_w1_s1,
//                                     out_w2_s1, out_w3_s1, output_w_idx_s1,
//                                     output_bh_idx, remain, 2);

    int2 pos_out1_0 = {output_w_idx_s1, output_bh_idx};
    output[pos_out1_0] = out_w0_s1;
    if (remain >= 2) {
        int2 pos_out1_1 = {output_w_idx_s1 + 1, output_bh_idx};
        output[pos_out1_1] = out_w1_s1;
    }
    if (remain >= 3) {
        int2 pos_out1_2 = {output_w_idx_s1 + 2, output_bh_idx};
        output[pos_out1_2] = out_w2_s1;
    }
    if (remain >= 4) {
        int2 pos_out1_3 = {output_w_idx_s1 + 3, output_bh_idx};
        output[pos_out1_3] = out_w3_s1;
    }

}
