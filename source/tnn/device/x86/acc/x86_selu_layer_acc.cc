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
#include <cmath>

namespace TNN_NS {
typedef struct x86_selu_operator : x86_unary_operator {
public:
    Status Init(LayerParam *param) {
        auto layer_param = dynamic_cast<SeluLayerParam *>(param);
        if (!layer_param) {
            LOGE("Error: selu layer param is nil\n");
            return Status(TNNERR_MODEL_ERR, "Error: selu layer param is nil");
        }
        alpha_ = layer_param->alpha;
        gamma_ = layer_param->gamma;
        return TNN_OK;
    }
    virtual float operator()(const float v) {
        float tmp = v;
        if (tmp <= 0.f) {
            tmp = gamma_ * (alpha_ * exp(tmp) - alpha_);
        } else {
            tmp = gamma_ * tmp;
        }
        return tmp;
    }

private:
    float alpha_ = 0.f;
    float gamma_ = 0.f;
} X86_SELU_OP;

DECLARE_X86_UNARY_ACC(Selu, X86_SELU_OP);

REGISTER_X86_ACC(Selu, LAYER_SELU);

}   // namespace TNN_NS