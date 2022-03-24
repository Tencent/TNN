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

ByteAddressBuffer in_buf: register(t0);
RWByteAddressBuffer out_buf: register(u0);

cbuffer CB : register( b0 )
{
    unsigned int c_n;
    unsigned int c_c;
    unsigned int c_h;
    unsigned int c_w;
};

#define THREADS_PER_BLOCK 128
#define ELE_PER_THREAD 4

[numthreads(THREADS_PER_BLOCK, 1, 1)]
void CSMain( uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint buf_len;
    buf_len = c_n * c_c * c_h * c_w;
    

    [unroll(ELE_PER_THREAD)] for(uint i=0;i<ELE_PER_THREAD;i++) {
        uint idx = groupIndex.x + i * THREADS_PER_BLOCK +  ELE_PER_THREAD  * THREADS_PER_BLOCK * groupID.x;
        if (idx < buf_len) {
            float f0 = asfloat( in_buf.Load(idx * 4)) + 1.1f;
            out_buf.Store(idx * 4, asuint(f0));
        }
    }
    
}

