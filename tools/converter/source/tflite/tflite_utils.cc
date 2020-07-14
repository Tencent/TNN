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

#include "tf_lite_utils.h"

bool ConvertDataFormatTflite(const float* src, float* dst, int KH, int KW, int CI, int CO) {
    // DCHECK(KH > 0);
    // DCHECK(KW > 0);
    // DCHECK(CI > 0);
    // DCHECK(CO > 0);
    // DCHECK(src != nullptr);
    // CO KH KW CI --> CO CI KH KW
    for (int oc = 0; oc < CO; ++oc) {
        for (int ic = 0; ic < CI; ++ic) {
            for (int h = 0; h < KH; ++h) {
                for (int w = 0; w < KW; ++w) {
                    dst[(oc * CI + ic) * KH * KW + h * KW + w] = src[(oc * KH + h) * KW * CI + w * CI + ic];
                }
            }
        }
    }

    return true;
}