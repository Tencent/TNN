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

#define THREADS_PER_BLOCK 64
#define THREAD_BLOCK_A 2
#define THREAD_BLOCK_B 8
#define BLOCK_A 16
#define BLOCK_B 64

#define WARP_SIZE 32
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / WARP_SIZE)
#define THREADS_PER_GROUP (BLOCK_A / (WARPS_PER_BLOCK * THREAD_BLOCK_A))

#define SMEM_A UP_DIV(BLOCK_A * 4, WARP_SIZE) * WARP_SIZE
#define SMEM_B UP_DIV(BLOCK_B, WARP_SIZE) * WARP_SIZE

groupshared float a_mem[4][SMEM_A];
groupshared float b_mem[4][SMEM_B];

[numthreads(THREADS_PER_BLOCK, 1, 1)]
void CSMain( uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex )
{
    uint oc4 = UP_DIV(out_shape[1], 4);
    uint oh  = out_shape[2];
    uint ow  = out_shape[3];

    uint M = oh * ow;
    uint K = in_shape[1];
    uint N = out_shape[1];

    uint b_m64_idx = groupID.y;
    uint b_idx     = b_m64_idx / (UP_DIV(M, 64));
    uint m64_idx   = b_m64_idx % (UP_DIV(M, 64));

    uint load_offset = b_idx * oh;

    uint a_offset_y = groupID.x * BLOCK_A + groupIndex.x / 4;
    uint a_offset_x = groupIndex.x % 4;
    uint b_offset   = m64_idx * BLOCK_B + groupIndex.x;
    uint b_offset_w = b_offset % ow;
    uint b_offset_h = b_offset / ow;

    const uint warp_id = groupIndex.x / WARP_SIZE;
    const uint lane_id = groupIndex.x % WARP_SIZE;
    const uint swizzledA = (warp_id * THREADS_PER_GROUP + lane_id % THREADS_PER_GROUP ) * THREAD_BLOCK_A;
    const uint swizzledB = lane_id / THREADS_PER_GROUP * THREAD_BLOCK_B;

    const uint c_off_a = BLOCK_A * groupID.x + swizzledA;
    const uint c_off_b = BLOCK_B * m64_idx + swizzledB;

    uint oh_offset[8];
    uint ow_offset[8];
    uint oh_idx = c_off_b / ow;
    uint ow_idx = c_off_b % ow;
    for (int i = 0; i < 8; ++i) {
        if (ow_idx >= ow) {
            ow_idx = 0;
            oh_idx += 1;
        }
        ow_offset[i] = ow_idx;
        oh_offset[i] = oh_idx;
        ow_idx += 1;
    }

    float4 a_rf[4][THREAD_BLOCK_A];
    float4 b_rf[THREAD_BLOCK_B];
    float4 c_rf[THREAD_BLOCK_B][THREAD_BLOCK_A];

    [unroll]  for (uint j = 0; j < THREAD_BLOCK_B; ++j) {
        [unroll]  for(uint i = 0; i < THREAD_BLOCK_A; ++i) {
            uint2 bias_pos = {c_off_a + i, 0};
            c_rf[j][i] = bias[bias_pos];
        }
    }

    for (uint k = 0; k < UP_DIV(K, 4); ++k) {
        uint2 pos_b = {k * ow + b_offset_w, load_offset + b_offset_h};
        b_mem[0][groupIndex.x] = input[pos_b].x;
        b_mem[1][groupIndex.x] = input[pos_b].y;
        b_mem[2][groupIndex.x] = input[pos_b].z;
        b_mem[3][groupIndex.x] = input[pos_b].w;

        uint2 pos_a = {k * 4 + a_offset_x, a_offset_y};
        a_mem[0][groupIndex.x] = weights[pos_a].x;
        a_mem[1][groupIndex.x] = weights[pos_a].y;
        a_mem[2][groupIndex.x] = weights[pos_a].z;
        a_mem[3][groupIndex.x] = weights[pos_a].w;

        GroupMemoryBarrierWithGroupSync();

        // smem to register file
        [unroll] for(uint a_of=0;a_of<THREAD_BLOCK_A;a_of++){
            [unroll] for(uint a_of_y=0;a_of_y<4;a_of_y++) {
                float4 va = {a_mem[0][(swizzledA + a_of)*4 + a_of_y],
                             a_mem[1][(swizzledA + a_of)*4 + a_of_y],
                             a_mem[2][(swizzledA + a_of)*4 + a_of_y],
                             a_mem[3][(swizzledA + a_of)*4 + a_of_y]};
                a_rf[a_of_y][a_of] = va;
            }
        }
        [unroll] for(uint b_of=0;b_of<THREAD_BLOCK_B;b_of++){
            float4 vb = {b_mem[0][swizzledB+b_of],
                         b_mem[1][swizzledB+b_of],
                         b_mem[2][swizzledB+b_of],
                         b_mem[3][swizzledB+b_of]};
            b_rf[b_of] = vb;
        }

        // calc out product of size BLOCK_A * BLOCK_B
        [unroll]  for(uint a_of=0;a_of<THREAD_BLOCK_A;a_of++) {
            [unroll]  for(uint b_of=0;b_of<THREAD_BLOCK_B;b_of++) {
                c_rf[b_of][a_of] += a_rf[0][a_of] * b_rf[b_of].x;
                c_rf[b_of][a_of] += a_rf[1][a_of] * b_rf[b_of].y;
                c_rf[b_of][a_of] += a_rf[2][a_of] * b_rf[b_of].z;
                c_rf[b_of][a_of] += a_rf[3][a_of] * b_rf[b_of].w;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    for (uint j = 0; j < 8; ++j) {
        for(uint i = 0; i < 2; ++i){
            uint2 dst_pos = {(c_off_a + i) * ow + ow_offset[j], load_offset + oh_offset[j]};
            if (oh_offset[j] < oh && c_off_a + i < oc4) {
                ActivationProcess(c_rf[j][i], activation_type[0]);
                output[dst_pos] = c_rf[j][i];
            }
        }
    }
}
