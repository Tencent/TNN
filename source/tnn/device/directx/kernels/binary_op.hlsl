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

ByteAddressBuffer in_a: register(t0);
ByteAddressBuffer in_b: register(t1);
RWByteAddressBuffer out_c: register(u0);

struct StepsAndShapes {

};

cbuffer StepsAndShapes: register( b0 )
{
    // NB 
    // in a constant buffer, each element of an array must start on a 4-float boundary. 
    // so we choose float4 for the ease of alignment with cpp

    // input a steps
    vector<uint, 4> as_0;
    vector<uint, 4> as_3;

    // input b steps
    vector<uint, 4> bs_0;
    vector<uint, 4> bs_3;

    // output dimension 
    vector<uint, 4> od_0;
    vector<uint, 4> od_3;
};

#ifndef BINARY_OP
#define BINARY_OP -
#endif


#define THREADS_PER_BLOCK 128
#define ELE_PER_THREAD 4

[numthreads(THREADS_PER_BLOCK, 1, 1)]
void CSMain( uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint buf_len = 1;
    [unroll(3)] for(uint i=0; i < 3; i++) {
        buf_len *= od_0[i];
        buf_len *= od_3[i];
    }

    [unroll(ELE_PER_THREAD)] for(uint i=0;i<ELE_PER_THREAD;i++) {
        uint idx = groupIndex.x + i * THREADS_PER_BLOCK +  ELE_PER_THREAD  * THREADS_PER_BLOCK * groupID.x;
        if (idx < buf_len) {
            vector<uint, 4> ic_0; 
            vector<uint, 4> ic_3; 

            uint remains = idx;
            uint idx_a = 0;
            uint idx_b = 0;

            [unroll(3)] for(uint i=2;i>=0;i--){
                ic_3[i] = remains % od_3[i];
                remains /= od_3[i];
                idx_a += ic_3[i] * as_3[i];
                idx_b += ic_3[i] * bs_3[i];
            }

            [unroll(3)] for(uint i=2;i>=0;i--){
                ic_0[i] = remains % od_0[i];
                remains /= od_0[i];
                idx_a += ic_0[i] * as_0[i];
                idx_b += ic_0[i] * bs_0[i];
            }

            float fa = asfloat( in_a.Load(idx_a * 4));
            float fb = asfloat( in_b.Load(idx_b * 4));
            out_c.Store(idx * 4, asuint(fa BINARY_OP fb));
        }
    }
    
}

