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

[numthreads(1, 1, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{

    uint data0 = asuint( Buffer0.Load( DTid.x*4 ) );
  
	float b0, g0, r0, a0;

	a0 =  data0 >> 24;
	r0 = (data0 & 0x00ff0000) >> 16;
	g0 = (data0 & 0x0000ff00) >> 8;
	b0 = (data0 & 0x000000ff);

    BufferOut.Store( (DTid.x)*4, asuint(b0*scale0+bias0) );
    BufferOut.Store( (h*w + DTid.x)*4, asuint(g0*scale1+bias1) );
    BufferOut.Store( (h*w*2 + DTid.x)*4, asuint(r0*scale2+bias2) );
    BufferOut.Store( (h*w*3 + DTid.x)*4, asuint(a0*scale3+bias3) );

}
