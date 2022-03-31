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

cbuffer InputCBBuffer : register(b0)
{
     float scale0;
     float scale1;
     float scale2;
     float scale3;

     float bias0;
     float bias1;
     float bias2;
     float bias3;

     int n;
     int c;
     int h;
     int w;
};

ByteAddressBuffer Buffer0 : register(t0);
RWByteAddressBuffer BufferOut : register(u0);

#define THREADS_PER_BLOCK 128
#define ELE_PER_THREAD 1
#define STRIDE 4

[numthreads(THREADS_PER_BLOCK, 1, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    uint hw0 = h*w*0;
    uint hw1 = h*w*1;
    uint hw2 = h*w*2;
    uint hw3 = h*w*3;
    uint zchw = DTid.z*c*h*w;

    uint data0 = asuint( Buffer0.Load( (DTid.x + zchw)*STRIDE ) );
  
	float b0, g0, r0, a0;

	a0 =  data0 >> 24;
	r0 = (data0 & 0x00ff0000) >> 16;
	g0 = (data0 & 0x0000ff00) >> 8;
	b0 = (data0 & 0x000000ff);

    BufferOut.Store( (hw0 + DTid.x + zchw)*STRIDE, asuint(b0*scale0+bias0) );
    BufferOut.Store( (hw1 + DTid.x + zchw)*STRIDE, asuint(g0*scale1+bias1) );
    BufferOut.Store( (hw2 + DTid.x + zchw)*STRIDE, asuint(r0*scale2+bias2) );
    BufferOut.Store( (hw3 + DTid.x + zchw)*STRIDE, asuint(a0*scale3+bias3) );

}
