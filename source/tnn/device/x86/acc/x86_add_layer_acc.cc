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

#include "tnn/device/x86/acc/x86_binary_op_layer_acc.h"

namespace TNN_NS {

typedef struct x86_add_layer_operator : x86_binary_operator {
    virtual float operator()(const float v1, const float v2) {
        return v1 + v2;
    }
} X86_ADD_OP;

DECLARE_X86_BINARY_OP_ACC(Add, X86_ADD_OP);

REGISTER_X86_ACC(Add, LAYER_ADD);

}   // namespace TNN_NS