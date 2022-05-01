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


Texture2D<float4> matrix_m : register(t0);
Texture2D<float4> bias : register(t1);
RWTexture2D<float4> output : register(u0);

[numthreads(4, 4, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{

    int output_cw_idx = DTid.x; //c/4 w/2
    int output_bh_idx = DTid.y; //b h/2

    int out_channel = out_shape[1];
    int out_height = out_shape[2];
    int out_width = out_shape[3];
    int round_h = UP_DIV(out_shape[2], 2);
    int round_w = UP_DIV(out_shape[3], 2);
    int output_channel_blocks = UP_DIV(out_channel, 4);
    int global_size_dim1 = out_shape[0]*round_h;

    if (output_cw_idx >= output_channel_blocks*round_w || output_bh_idx >= out_shape[0]*round_h) {
        return;
    }
    int c_block_idx = output_cw_idx / round_w;
    int w_block_idx = output_cw_idx - mul(c_block_idx, round_w);
    int batch = output_bh_idx / round_h;
    int h_block_idx = output_bh_idx - mul(batch, round_h);

    int2 pos_bias = {c_block_idx, 0};
    float4 bias_value = bias[pos_bias];

    int2 pos_m00 = {output_cw_idx, output_bh_idx};
    float4 m00 = matrix_m[pos_m00];
    int2 pos_m10 = {output_cw_idx, mad(1, global_size_dim1, output_bh_idx)};
    float4 m10 = matrix_m[pos_m10];
    int2 pos_m20 = {output_cw_idx, mad(2, global_size_dim1, output_bh_idx)};
    float4 m20 = matrix_m[pos_m20];
    int2 pos_m30 = {output_cw_idx, mad(3, global_size_dim1, output_bh_idx)};
    float4 m30 = matrix_m[pos_m30];
    int2 pos_m01 = {output_cw_idx, mad(4, global_size_dim1, output_bh_idx)};
    float4 m01 = matrix_m[pos_m01];
    int2 pos_m11 = {output_cw_idx, mad(5, global_size_dim1, output_bh_idx)};
    float4 m11 = matrix_m[pos_m11];
    int2 pos_m21 = {output_cw_idx, mad(6, global_size_dim1, output_bh_idx)};
    float4 m21 = matrix_m[pos_m21];
    int2 pos_m31 = {output_cw_idx, mad(7, global_size_dim1, output_bh_idx)};
    float4 m31 = matrix_m[pos_m31];
    int2 pos_m02 = {output_cw_idx, mad(8, global_size_dim1, output_bh_idx)};
    float4 m02 = matrix_m[pos_m02];
    int2 pos_m12 = {output_cw_idx, mad(9, global_size_dim1, output_bh_idx)};
    float4 m12 = matrix_m[pos_m12];
    int2 pos_m22 = {output_cw_idx, mad(10, global_size_dim1, output_bh_idx)};
    float4 m22 = matrix_m[pos_m22];
    int2 pos_m32 = {output_cw_idx, mad(11, global_size_dim1, output_bh_idx)};
    float4 m32 = matrix_m[pos_m32];
    int2 pos_m03 = {output_cw_idx, mad(12, global_size_dim1, output_bh_idx)};
    float4 m03 = matrix_m[pos_m03];
    int2 pos_m13 = {output_cw_idx, mad(13, global_size_dim1, output_bh_idx)};
    float4 m13 = matrix_m[pos_m13];
    int2 pos_m23 = {output_cw_idx, mad(14, global_size_dim1, output_bh_idx)};
    float4 m23 = matrix_m[pos_m23];
    int2 pos_m33 = {output_cw_idx, mad(15, global_size_dim1, output_bh_idx)};
    float4 m33 = matrix_m[pos_m33];

    float4 out00  = m00 + m01 + m02;
    float4 out10  = m10 + m11 + m12;
    float4 out20  = m20 + m21 + m22;
    float4 out30  = m30 + m31 + m32;
    float4 out01  = m01 - m02 + m03;
    float4 out11  = m11 - m12 + m13;
    float4 out21  = m21 - m22 + m23;
    float4 out31  = m31 - m32 + m33;

    int2 ow = (int2)(w_block_idx << 1) + (int2)(0, 1);
    int2 oh = (int2)(h_block_idx << 1) + (int2)(0, 1);
    int2 ox = mad((int2)(c_block_idx), (int2)(out_width), ow);
    int2 oy = mad((int2)(batch), (int2)(out_height), oh);
    float4 res00  = bias_value + out00 + out10 + out20;
    res00 = ActivationProcess(res00, activation_type[0]);

    int2 pos_res00 = {ox.x, oy.x};
    output[pos_res00] = res00;
    if (ow.y < out_width && oh.x < out_height) {
        float4 res10  = bias_value + out10 - out20 + out30;
        res10 = ActivationProcess(res10, activation_type[0]);
        int2 pos_res10 = {ox.y, oy.x};
        output[pos_res10] = res10;
    }
    if (ow.x < out_width && oh.y < out_height) {
        float4 res01  = bias_value + out01 + out11 + out21;
        res01 = ActivationProcess(res01, activation_type[0]);
        int2 pos_res01 = {ox.x, oy.y};
        output[pos_res01] = res01;
    }
    if (ow.y < out_width && oh.y < out_height) {
        float4 res11  = bias_value + out11 - out21 + out31;
        res11 = ActivationProcess(res11, activation_type[0]);
        int2 pos_res11 = {ox.y, oy.y};
        output[pos_res11] = res11;
    }

}
