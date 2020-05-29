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

#include "tnn/utils/bfp16_utils.h"

#include <stdint.h>

#include "tnn/core/macro.h"
#include "tnn/utils/bfp16.h"

namespace TNN_NS {

int ConvertFromFloatToBFP16(float *fp32, void *fp16, int count) {
    bfp16_t *bfp16PTR = (bfp16_t *)fp16;
    for (int i = 0; i < count; ++i) {
        bfp16PTR[i] = fp32[i];
    }

    return 0;
}

int ConvertFromBFP16ToFloat(void *fp16, float *fp32, int count) {
    bfp16_t *bfp16PTR = (bfp16_t *)fp16;
    for (int i = 0; i < count; ++i) {
        fp32[i] = float(bfp16PTR[i]);
    }

    return 0;
}

}  // namespace TNN_NS
