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

Texture2D<float4> input : register(t0);
Texture2D<float4> weights : register(t1);
Texture2D<float4> bias : register(t2);
RWTexture2D<float4> output : register(u0);

cbuffer Shapes: register( b0 )
{
    // NB
    // in a constant buffer, each element of an array must start on a 4-float boundary.
    // so we choose float4 for the ease of alignment with cpp

    // output dimension
    //N, M, K
    vector<int, 4> innerproduct_shape;

};

[numthreads(64, 1, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    int col = DTid.x;
    int row = DTid.y;
    int N = innerproduct_shape[0];
    int M = innerproduct_shape[1];
    int K = innerproduct_shape[2];

    int K_blocks = UP_DIV(K, 4);
    int K_remain = K % 4;

    if (col >= UP_DIV(N, 4) || row >= M) {
        return;
    }

    float4 data;
    float4 weight_0;
    float4 weight_1;
    float4 weight_2;
    float4 weight_3;
    float4 bias_;
    float4 sum = {0, 0, 0, 0};

    int k_size = K;
    if (K_remain > 0) {
        k_size--;
    }

    int y = 0;

    for (int i = 0; i < k_size; i++) {
        y = i << 2;
        int2 pos_data0 = {i, row};
        data = input[pos_data0];
        int2 pos_w0 = {col, y};
        weight_0 = weights[pos_w0];
        int2 pos_w1 = {col, y + 1};
        weight_1 = weights[pos_w1];
        int2 pos_w2 = {col, y + 2};
        weight_2 = weights[pos_w2];
        int2 pos_w3 = {col, y + 3};
        weight_3 = weights[pos_w3];

        sum = mad(data.x, weight_0, sum);
        sum = mad(data.y, weight_1, sum);
        sum = mad(data.z, weight_2, sum);
        sum = mad(data.w, weight_3, sum);
    }

    if (K_remain > 0) {

        int2 pos_data1 = {i, row};
        data = input[pos_data1];
        y = i << 2;
    }

    if(K_remain == 3) {
        int2 pos_w0 = {col, y};
        weight_0 = weights[pos_w0];
        int2 pos_w1 = {col, y + 1};
        weight_1 = weights[pos_w1];
        int2 pos_w2 = {col, y + 2};
        weight_2 = weights[pos_w2];
        sum = mad(data.x, weight_0, sum);
        sum = mad(data.y, weight_1, sum);
        sum = mad(data.z, weight_2, sum);
    } if(K_remain == 2) {
        int2 pos_w0 = {col, y};
        weight_0 = weights[pos_w0];
        int2 pos_w1 = {col, y + 1};
        weight_1 = weights[pos_w1];
        sum = mad(data.x, weight_0, sum);
        sum = mad(data.y, weight_1, sum);
    } if(K_remain == 1) {
        int2 pos_w0 = {col, y};
        weight_0 = weights[pos_w0];
        sum = mad(data.x, weight_0, sum);
    }

    int2 pos_bias = {col, 0};
    bias_ = bias[pos_bias];
    sum += bias_;

    int2 pos_out = {col, row};
    output[pos_out] = sum;

}