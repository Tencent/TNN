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

Texture2D<float4> input: register(t0);
RWTexture2D<float4> output: register(u0);

cbuffer StepsAndShapes: register( b0 )
{
    // NB
    // in a constant buffer, each element of an array must start on a 4-float boundary.
    // so we choose float4 for the ease of alignment with cpp

    // input dimension
    vector<int, 4> id;

    // output dimension
    vector<int, 4> od;

    // pad_wh
    vector<int, 2> pad_wh;

    // stride_wh
    vector<int, 2> stride_wh;

    // kernel_wh
    vector<int, 2> kernel_wh;

};


[numthreads(1, 1, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID)
{
    int output_channel_idx      = DTid.z;
    int output_width_idx        = DTid.y;
    int output_batch_height_idx = DTid.x;

    if (output_channel_idx >= od[1]  || output_width_idx >= od[3] || output_batch_height_idx >= od[0]*od[2]) {
        return;
    }

    int output_width = od[3];
    int output_height = od[2];

    int output_batch_idx    = output_batch_height_idx / output_height;
    int output_height_idx   = output_batch_height_idx - mul(output_batch_idx, output_height);
    int input_start         = mul(output_batch_idx, id[2]);
    int input_height_start  = mad(output_height_idx, stride_wh[1], -pad_wh[1]);
    int input_width_start   = mad(output_width_idx, stride_wh[0], -pad_wh[0]);
    int input_channel_start = mul(output_channel_idx, id[3]);

    float4 output_result = 0;
      for (int height = 0; height < kernel_wh[1]; height++) {
          int input_height_idx = input_height_start + height;
          if(input_height_idx < 0 || input_height_idx >= id[2]){
            input_height_idx = -1;
          } else {
            input_height_idx = input_start + input_height_idx;
          }
          for (int width = 0; width < kernel_wh[0]; width++) {
              int input_width_idx = input_width_start + width;
              if(input_width_idx < 0 || input_width_idx >= id[3]) {
                input_width_idx = -1;
              } else {
                input_width_idx = input_channel_start + input_width_idx;
              }

              uint2 pos_in = {input_width_idx, input_height_idx};
              float4 input_data = input[pos_in];
              output_result     = output_result + input_data;
          }
      }

      int kernel_height_start = max(0, input_height_start);
      int kernel_width_start  = max(0, input_width_start);
      int kernel_height_end   = min(input_height_start + kernel_wh[1], id[2]);
      int kernel_width_end    = min(input_width_start + kernel_wh[0], id[3]);
      int block_size = mul((kernel_height_end - kernel_height_start), (kernel_width_end - kernel_width_start));
      output_result = output_result / (float)block_size;

      uint output_channel_width_idx = mad(output_channel_idx, output_width, output_width_idx);
      uint2 pos_out = {output_channel_width_idx, output_batch_height_idx};
      output[pos_out] = output_result;
}

