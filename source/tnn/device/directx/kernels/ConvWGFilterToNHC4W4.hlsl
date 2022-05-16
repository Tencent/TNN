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
#define PACK4 4
#define STRIDE 4

ByteAddressBuffer input : register(t0);
RWTexture2D<float4> output : register(u0);

[numthreads(4, 4, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    int x = DTid.x;
    int y = DTid.y;

    int width, height;
    output.GetDimensions(width, height);

    if (x < width && y < height) {
        float4 out4;
        out4.x =asfloat(input.Load(((x + y * width)*PACK4 + 0)*STRIDE));
        out4.y =asfloat(input.Load(((x + y * width)*PACK4 + 1)*STRIDE));
        out4.z =asfloat(input.Load(((x + y * width)*PACK4 + 2)*STRIDE));
        out4.w =asfloat(input.Load(((x + y * width)*PACK4 + 3)*STRIDE));

        output[DTid.xy] = out4;
    }

}