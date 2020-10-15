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
#include <math.h>

namespace TNN_NS {

typedef struct x86_hardswish_layer_acc : x86_unary_operator {
public:
    Status Init(LayerParam *param) {
        auto layer_param = dynamic_cast<HardSwishLayerParam *>(param);
        if (!layer_param) {
            LOGE("Error: clip layer param is nil\n");
            return Status(TNNERR_MODEL_ERR, "Error, hard swish layer param is nil");
        }
        alpha_ = layer_param->alpha;
        beta_  = layer_param->beta;
        return TNN_OK;
    }

    virtual float operator()(const float v) {
        float tmp = v * alpha_ + beta_;
        tmp = std::min(1.0f, tmp);
        tmp = std::max(0.0f, tmp);
        tmp *= v;
        return tmp;
    }
private:
    float min_ = 0.f;
    float max_ = 0.f;
    float alpha_ = 1.f;
    float beta_ = 0.f;
} X86_HARDSWISH_OP;

DECLARE_X86_UNARY_ACC(HardSwish, X86_HARDSWISH_OP);

// REGISTER_X86_ACC(HardSwish, LAYER_HARDSWISH);

}   // namespace TNN_NS
