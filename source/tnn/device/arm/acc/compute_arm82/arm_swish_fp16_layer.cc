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

#include "tnn/device/arm/acc/compute_arm82/arm_unary_fp16_layer.h"

namespace TNN_NS {

#if TNN_ARM82
typedef struct arm_swish_fp16_operator {
    virtual Half8 operator()(const Half8 &v) {
        return v * Half8::sigmoid(v);
    }
} ARM_SWISH_OP;

DEFINE_ARM_UNARY_ACC_FP16(Swish, ARM_SWISH_OP);
#endif

}  // namespace TNN_NS
