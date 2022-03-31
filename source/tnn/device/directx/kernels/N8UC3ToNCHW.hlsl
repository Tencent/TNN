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

ByteAddressBuffer BufferIn : register(t0);
RWByteAddressBuffer BufferOut : register(u0);

#define THREADS_PER_BLOCK 128
#define ELE_PER_THREAD 4
#define STRIDE 4

[numthreads(THREADS_PER_BLOCK, 1, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    // DTid.x <- c*h*w
    // DTid.z <- n
    uint hw0 = h*w*0;
    uint hw1 = h*w*1;
    uint hw2 = h*w*2;
    uint xc = DTid.x*c;
    uint zchw = DTid.z*c*h*w;
    uint x4 = DTid.x*4;
    uint nchw = n*c*h*w;

    uint data0 = asuint( BufferIn.Load( (xc + 0 + zchw)*STRIDE) );
	uint data1 = asuint( BufferIn.Load( (xc + 1 + zchw)*STRIDE) );
	uint data2 = asuint( BufferIn.Load( (xc + 2 + zchw)*STRIDE) );

	float b0, g0, r0;
	float b1, g1, r1;
	float b2, g2, r2;
	float b3, g3, r3;

	b1 =  data0 >> 24;
	r0 = (data0 & 0x00ff0000) >> 16;
	g0 = (data0 & 0x0000ff00) >> 8;
	b0 = (data0 & 0x000000ff);

	g2 =  data1 >> 24;
	b2 = (data1 & 0x00ff0000) >> 16;
	r1 = (data1 & 0x0000ff00) >> 8;
	g1 = (data1 & 0x000000ff);

	r3 =  data2 >> 24;
	g3 = (data2 & 0x00ff0000) >> 16;
	b3 = (data2 & 0x0000ff00) >> 8;
	r2 = (data2 & 0x000000ff);

    BufferOut.Store( (hw0 + x4 + 0 + zchw)*STRIDE, asuint(b0*scale0+bias0) );
    BufferOut.Store( (hw1 + x4 + 0 + zchw)*STRIDE, asuint(g0*scale1+bias1) );
    BufferOut.Store( (hw2 + x4 + 0 + zchw)*STRIDE, asuint(r0*scale2+bias2) );

	BufferOut.Store( (hw0 + x4 + 1 + zchw)*STRIDE, asuint(b1*scale0+bias0) );
    BufferOut.Store( (hw1 + x4 + 1 + zchw)*STRIDE, asuint(g1*scale1+bias1) );
    BufferOut.Store( (hw2 + x4 + 1 + zchw)*STRIDE, asuint(r1*scale2+bias2) );

	BufferOut.Store( (hw0 + x4 + 2 + zchw)*STRIDE, asuint(b2*scale0+bias0) );
    BufferOut.Store( (hw1 + x4 + 2 + zchw)*STRIDE, asuint(g2*scale1+bias1) );
    BufferOut.Store( (hw2 + x4 + 2 + zchw)*STRIDE, asuint(r2*scale2+bias2) );

	BufferOut.Store( (hw0 + x4 + 3 + zchw)*STRIDE, asuint(b3*scale0+bias0) );
    BufferOut.Store( (hw1 + x4 + 3 + zchw)*STRIDE, asuint(g3*scale1+bias1) );
    BufferOut.Store( (hw2 + x4 + 3 + zchw)*STRIDE, asuint(r3*scale2+bias2) );
}
