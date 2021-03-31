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

#include "tnn/device/x86/acc/x86_unary_layer_acc.h"
#include "tnn/interpreter/layer_param.h"

#include <cmath>

namespace TNN_NS {

typedef struct x86_power_operator : x86_unary_operator {
public:
    Status Init(LayerParam *param) {
        auto layer_param = dynamic_cast<PowLayerParam *>(param);
        if (!layer_param) {
            LOGE("Error: pow layer param is nil\n");
            return Status(TNNERR_MODEL_ERR, "Error: pow layer param is nil");
        }
        scale_      = layer_param->scale;
        shift_      = layer_param->shift;
        exponent_   = layer_param->exponent;
        return TNN_OK;
    }

    virtual float operator()(const float v) {
        return pow(v * scale_ + shift_, exponent_);
    }

private:
    float scale_ = 1.f;
    float shift_ = 0.f;
    float exponent_ = 0.f;
} X86_POW_OP;

DECLARE_X86_UNARY_ACC(Pow, X86_POW_OP);

REGISTER_X86_ACC(Pow, LAYER_POWER);

}   // namespace TNN_NS