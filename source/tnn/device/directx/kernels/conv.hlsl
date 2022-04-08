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

ByteAddressBuffer in_buf: register(t0);     // NCHW
ByteAddressBuffer weight_buf: register(t1); // OIHW
ByteAddressBuffer bias_buf: register(t2); // Out_C

ByteAddressBuffer ptr_offset: register(t3); // OH * OW
ByteAddressBuffer load_mask: register(t4); // OH * OW, 9 bits indicates R * S in image or not
ByteAddressBuffer filter_offset: register(t5); // 9 // load offset of crs + LOOP_K on each [R*S] pos
ByteAddressBuffer warp_offset: register(t6); // 4 // load offset of crs+1
RWByteAddressBuffer out_buf: register(u0);  // NCHW

cbuffer StepsAndShapes: register( b0 )
{
    // NB 
    // in a constant buffer, each element of an array must start on a 4-float boundary. 
    // so we choose float4 for the ease of alignment with cpp

    // input a shape
    vector<uint, 4> in_nchw;
    vector<uint, 4> out_nchw;
    vector<uint, 4> filter_kcrs;
    vector<uint, 4> fused_relu;

};

#define WARP_SIZE 32

#ifndef BLOCK_A
#define BLOCK_A 64
#endif

#define UP_DIV(A, B) (((A) + (B) - 1) / B)

#define THREADS_PER_BLOCK 64
#define LOOP_K 4
#define THREAD_BLOCK_A 8
#define THREAD_BLOCK_B 8
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / WARP_SIZE)
#define THREADS_PER_GROUP (BLOCK_A / (WARPS_PER_BLOCK * THREAD_BLOCK_A))
#define BLOCK_B ((32/THREADS_PER_GROUP) * (THREAD_BLOCK_B))
#define SMEM_A UP_DIV(BLOCK_A, WARP_SIZE) * WARP_SIZE
#define SMEM_B UP_DIV(BLOCK_B, WARP_SIZE) * WARP_SIZE

#define ReadInt(_buf, index) asint(_buf.Load((index) * 4))
#define ReadUint(_buf, index) asuint(_buf.Load((index) * 4))
#define ReadFloat(_buf, index) asfloat(_buf.Load((index) * 4))
#define WriteFloat(_buf, index, value) _buf.Store((index) * 4, asuint(value))

groupshared float a_mem[LOOP_K][SMEM_A];
groupshared float b_mem[LOOP_K][SMEM_B];

[numthreads(THREADS_PER_BLOCK, 1, 1)]
void CSMain( uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // TODO Support multi-batch

    const uint IC = in_nchw[1];
    const uint OC = out_nchw[1];
    const uint RS = filter_kcrs[2] * filter_kcrs[3];
    const uint OHW = out_nchw[2] * out_nchw[3];
    const uint IN_NS = in_nchw[1] * in_nchw[2] * in_nchw[3];
    const uint OUT_NS = out_nchw[1] * out_nchw[2] * out_nchw[3];

    int a_row = groupID.x * BLOCK_A + groupIndex.x;
    int mask = ReadInt(load_mask, a_row);

    int a_offset[LOOP_K];
    int b_offset = groupID.y * BLOCK_B + groupIndex.x;

    for( uint i = 0; i< LOOP_K ; i++ ) {
        a_offset[i] = ReadInt(ptr_offset, a_row) + ReadInt(warp_offset, i);
    }


    const uint warp_id = groupIndex.x / WARP_SIZE;
    const uint lane_id = groupIndex.x % WARP_SIZE;

    const uint swizzledA = (warp_id * THREADS_PER_GROUP + lane_id % THREADS_PER_GROUP ) * THREAD_BLOCK_A; // line_id of A
    const uint swizzledB = lane_id / THREADS_PER_GROUP * THREAD_BLOCK_B; // line_id of B

    const uint c_off_a = BLOCK_A * groupID.x + swizzledA; 
    const uint c_off_b = BLOCK_B * groupID.y + swizzledB;  

    float a_rf[THREAD_BLOCK_A];
    float b_rf[THREAD_BLOCK_B];
    float c_rf[THREAD_BLOCK_A*THREAD_BLOCK_B] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // step 1. loop over CRS by LOOP_K
    uint col = 0;
    for(; col < IC*RS ;) {
       // step 2. load BLOCK_A x LOOP_K input and BLOCK_B x LOOP_K filter from global mem to shared mem 
        for(uint i=0;i<LOOP_K;i++){
            int bit = col % RS;
            bool in_image = ((mask & (1 << bit)) && (a_row < OHW) && (col < IC*RS) );
            uint idx_a = in_image ? a_offset[i] : 0;
            a_mem[i][groupIndex.x] = in_image ? ReadFloat(in_buf, idx_a) : 0.f;


            // load filter, TODO permute weights to IRSO for collapsed reading
            bool in_filter = col < IC*RS && b_offset < OC;
            uint idx_b = in_filter ? b_offset * IC * RS + col : 0;
            b_mem[i][groupIndex.x] = in_filter ? ReadFloat(weight_buf, idx_b) : 0.f;

            a_offset[i] += ReadInt(filter_offset, bit);
            col += 1;
        }


        GroupMemoryBarrierWithGroupSync();


        // calc for LOOP_K elements
        for(uint k=0;k<LOOP_K;k++) {
            // smem to register file
            [unroll] for(uint a_of=0;a_of<THREAD_BLOCK_A;a_of++){
                a_rf[a_of] = a_mem[k][swizzledA+a_of];
            }
            [unroll] for(uint b_of=0;b_of<THREAD_BLOCK_B;b_of++){
                b_rf[b_of] = b_mem[k][swizzledB+b_of];
            }

            // calc out product of size BLOCK_A * BLOCK_B 
            [unroll]  for(uint a_of=0;a_of<THREAD_BLOCK_A;a_of++) {
                [unroll]  for(uint b_of=0;b_of<THREAD_BLOCK_B;b_of++) {
                    c_rf[b_of * THREAD_BLOCK_A + a_of] = a_rf[a_of] * b_rf[b_of] + c_rf[b_of*THREAD_BLOCK_A+a_of];
                }
            }
        }

    }

    for(uint j=0; j<THREAD_BLOCK_B;j++) {
        for(uint i=0; i<THREAD_BLOCK_A; i++){

            bool in_image = (c_off_a + i < OHW) && (c_off_b + j < OC);
            float bias = in_image ? ReadFloat(bias_buf, c_off_b + j) : 0.f;
            float vc = c_rf[j * THREAD_BLOCK_A + i] + bias;

            if (fused_relu[0]) {
                vc = vc > 0.f ? vc : 0.f;
            }

            uint dst_of = (c_off_b + j) * OHW + c_off_a + i;
            if(in_image) {
                WriteFloat(out_buf, dst_of, vc);
            }
        }
    }


}
