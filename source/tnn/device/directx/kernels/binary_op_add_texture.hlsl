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

Texture2D<float4> Buffer0 : register(t0);
ByteAddressBuffer Buffer1 : register(t1);
RWTexture2D<float4> BufferOut : register(u0);

[numthreads(1, 1, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    float4 f0 = Buffer0[DTid.xy];
    float f = asfloat( Buffer1.Load( 0 ) );
    float4 f1 = {f,f,f,f};

    BufferOut[DTid.xy] = f0 + f1;
}

