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

#define WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, outWidthIdx, outHeightIdx, remain)  \
        if (remain >= 4) { \
            int2 pos_out0 = {outWidthIdx, outHeightIdx}; \
            output[pos_out0] = out0; \
            int2 pos_out1 = {outWidthIdx + 1, outHeightIdx}; \
            output[pos_out1] = out1; \
            int2 pos_out2 = {outWidthIdx + 2, outHeightIdx}; \
            output[pos_out2] = out2; \
            int2 pos_out3 = {outWidthIdx + 3, outHeightIdx}; \
            output[pos_out3] = out3; \
        } else if (remain == 3) { \
            int2 pos_out0 = {outWidthIdx, outHeightIdx}; \
            output[pos_out0] = out0; \
            int2 pos_out1 = {outWidthIdx + 1, outHeightIdx}; \
            output[pos_out1] = out1; \
            int2 pos_out2 = {outWidthIdx + 2, outHeightIdx}; \
            output[pos_out2] = out2; \
        } else if (remain == 2) { \
            int2 pos_out0 = {outWidthIdx, outHeightIdx}; \
            output[pos_out0] = out0; \
            int2 pos_out1 = {outWidthIdx + 1, outHeightIdx}; \
            output[pos_out1] = out1; \
        } else if (remain == 1) { \
            int2 pos_out0 = {outWidthIdx, outHeightIdx}; \
            output[pos_out0] = out0; \
        } \

#define ActivationType_None 0x0000
#define ActivationType_ReLU 0x0001
#define ActivationType_ReLU6 0x0002
#define ActivationType_SIGMOID_MUL 0x0100

#define ActivationProcess(output, activation_type) \
    output = (activation_type == ActivationType_ReLU) ? max(output, (float4)0) : \
    ((activation_type == ActivationType_ReLU6) ? clamp(output,(float4)0,(float4)6) : \
    ((activation_type == ActivationType_SIGMOID_MUL) ? mul(rcp((float4)1+exp(-output)),output) : \
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

    vector<uint, 4> kernel_wh_;

    vector<uint, 4> stride_wh_;

    vector<uint, 4> padding_wh_;

    vector<uint, 4> dilation_wh_;

    vector<uint, 4> activation_type_;

};


Texture2D<float4> input : register(t0);
Texture2D<float4> filter : register(t1);
Texture2D<float4> bias : register(t2);
RWTexture2D<float4> output : register(u0);

[numthreads(4, 4, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    int2 input_wh = {in_shape[3], in_shape[2]};
    int2 output_wh = {out_shape[3], out_shape[2]};
    int2 kernel_wh = {kernel_wh_[0], kernel_wh_[1]};
    int2 padding_wh = {padding_wh_[0], padding_wh_[1]};
    int2 dilation_wh = {dilation_wh_[0], dilation_wh_[1]};
    int2 stride_wh = {stride_wh_[0], stride_wh_[1]};
    int activation_type = activation_type_[0];

    int outChannelWidthIdx = DTid.x;
    int outHeightIdx       = DTid.y;

    if (outChannelWidthIdx > UP_DIV(out_shape[1], 4)*UP_DIV(out_shape[3], 4) || outHeightIdx > out_shape[0]*out_shape[2]) {
        return;
    }

    int ow4 = (output_wh.x + 3) / 4;
    int outChannelBlockIdx = outChannelWidthIdx / ow4;
    int outWidthBlockidx   = outChannelWidthIdx % ow4;

    int inChannelBlockIdx = outChannelBlockIdx;

    int2 pos_bias = {outChannelBlockIdx, 0};
    float4 out0 = bias[pos_bias];
    float4 out1 = out0;
    float4 out2 = out0;
    float4 out3 = out0;

    int in_width0 = mad(outWidthBlockidx, stride_wh.x << 2, -padding_wh.x);
    int in_width1 = in_width0 + stride_wh.x;
    int in_width2 = in_width1 + stride_wh.x;
    int in_width3 = in_width2 + stride_wh.x;
    int heightIdx = mad(outHeightIdx % output_wh.y, stride_wh.y, -padding_wh.y);

    int outBatchIdx = mul((outHeightIdx / output_wh.y), input_wh.y);

    int in_idx = mul(inChannelBlockIdx, input_wh.x);

    for (int kh = 0; kh < kernel_wh.y; kh++) {
        int in_hb_value = (heightIdx < 0 || heightIdx >= input_wh.y) ? -1 : heightIdx + outBatchIdx;
        heightIdx += dilation_wh.y;
        for (int kw = 0; kw < kernel_wh.x; kw++) {
            int filterIdx = mad(kh, kernel_wh.x, kw);
            float4 in0, in1, in2, in3;
            int inWidthIdx = mul(kw, dilation_wh.x);

            READ_INPUT_IMAGE(0, inWidthIdx);
            READ_INPUT_IMAGE(1, inWidthIdx);
            READ_INPUT_IMAGE(2, inWidthIdx);
            READ_INPUT_IMAGE(3, inWidthIdx);

            int2 pos_filter = {filterIdx, inChannelBlockIdx};
            float4 weights = filter[pos_filter];

            out0 = mad(in0, weights, out0);
            out1 = mad(in1, weights, out1);
            out2 = mad(in2, weights, out2);
            out3 = mad(in3, weights, out3);
        }
    }

    ActivationProcess(out0, activation_type);
    ActivationProcess(out1, activation_type);
    ActivationProcess(out2, activation_type);
    ActivationProcess(out3, activation_type);

    int outWidthBlockidx4 = outWidthBlockidx << 2;
    int remain            = output_wh.x - outWidthBlockidx4;
    int outWidthIdx = mul(outChannelBlockIdx, output_wh.x) + outWidthBlockidx4;

    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, outWidthIdx,
                               outHeightIdx, remain);

}
