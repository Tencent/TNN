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
RWByteAddressBuffer out_buf: register(u0);  // NCHW

cbuffer StepsAndShapes: register( b0 )
{
    // NB 
    // in a constant buffer, each element of an array must start on a 4-float boundary. 
    // so we choose float4 for the ease of alignment with cpp

    // input a shape
    vector<uint, 4> in_nchw;
    vector<uint, 4> out_nchw;

};

#define WARP_SIZE 32

#ifndef BLOCK_A
#define BLOCK_A 64
#endif

#define UP_DIV(A, B) (((A) + (B) - 1) / B)

#define THREADS_PER_BLOCK 128
#define LOOP_K 4
#define THREAD_BLOCK_A 4
#define THREAD_BLOCK_B 4
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / WARP_SIZE)
#define THREADS_PER_GROUP (BLOCK_A / (WARPS_PER_BLOCK * THREAD_BLOCK_A))
#define BLOCK_B ((32/THREADS_PER_GROUP) * (THREAD_BLOCK_B))
#define SMEM_A UP_DIV(BLOCK_A, WARP_SIZE) * WARP_SIZE
#define SMEM_B UP_DIV(BLOCK_B, WARP_SIZE) * WARP_SIZE


groupshared float a_mem[LOOP_K][SMEM_A];
groupshared float b_mem[LOOP_K][SMEM_B];

[numthreads(THREADS_PER_BLOCK, 1, 1)]
void CSMain( uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint IC = in_nchw[1];
    const uint OC = out_nchw[1];
    const uint HW = in_nchw[2] * in_nchw[3];
    const uint NHW = in_nchw[0] * in_nchw[2] * in_nchw[3];
    const uint IN_NS = in_nchw[1] * in_nchw[2] * in_nchw[3];
    const uint OUT_NS = out_nchw[1] * out_nchw[2] * out_nchw[3];

    uint group_offset = groupID.x * BLOCK_A;


    const uint warp_id = groupIndex.x / WARP_SIZE;
    const uint lane_id = groupIndex.x % WARP_SIZE;

    const uint swizzledA = (warp_id * THREADS_PER_GROUP + lane_id % THREADS_PER_GROUP ) * THREAD_BLOCK_A; // line_id of A
    const uint swizzledB = lane_id / THREADS_PER_GROUP * THREAD_BLOCK_B; // line_id of B


    for(uint n_start=0;n_start<OC;n_start+=BLOCK_B){
        float a_rf[THREAD_BLOCK_A];
        float b_rf[THREAD_BLOCK_B];
        float c_rf[THREAD_BLOCK_A*THREAD_BLOCK_B] = {
            0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0};

        // split k 
        for(uint k_start=0;k_start<IC;k_start+=LOOP_K) {

            uint ic_id = warp_id + k_start;
            [unroll] for(uint i=0;i<UP_DIV(BLOCK_A, WARP_SIZE);i++) {
                uint nhw_id = group_offset + lane_id + i * WARP_SIZE;
                uint hw_id = nhw_id % HW;
                uint batch_id = nhw_id / HW; 
                bool in_image = (ic_id < IC) && (nhw_id < NHW );

                uint load_offset = in_image ? batch_id * IN_NS + hw_id + ic_id * HW : 0;
                float va = asfloat( in_buf.Load(load_offset * 4));
                a_mem[warp_id][lane_id + i * WARP_SIZE] = in_image ? va : 0.f;
            }

            // TODO Optimize: change to collapsed read
            [unroll] for(uint i=0;i<UP_DIV(BLOCK_B, WARP_SIZE);i++) {
                uint oc_id = n_start + lane_id + i * WARP_SIZE;
                bool in_filter = ic_id < IC && oc_id < OC;

                uint load_offset = in_filter ? oc_id * IC + warp_id + k_start : 0;
                float vb = asfloat( weight_buf.Load(load_offset * 4));
                b_mem[warp_id][lane_id + i * WARP_SIZE] = in_filter ? vb : 0.f;
            }

            GroupMemoryBarrierWithGroupSync();


            // calc for LOOP_K elements
            [unroll(2)] for(uint k=0;k<LOOP_K;k++) {
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

            GroupMemoryBarrierWithGroupSync();
        }

        // TODO Optimize: change to collapsed store 
        for(uint a_of=0;a_of<THREAD_BLOCK_A;a_of++) {

            uint nhw_id = group_offset + swizzledA + a_of;
            uint n_id = nhw_id  / HW;
            uint hw_id =nhw_id  % HW;

            for(uint i=0;i<THREAD_BLOCK_B;i++){

                float vc = c_rf[i*THREAD_BLOCK_A + a_of];

                uint oc_id = n_start + swizzledB + i;
                uint store_index = n_id * OUT_NS + oc_id * HW + hw_id;
                if (nhw_id < NHW && oc_id < OC ) {
                    float vb = asfloat(bias_buf.Load(oc_id * 4));
                    out_buf.Store(store_index * 4, asuint(vc + vb));
                }
            }
        }  
    }
}

