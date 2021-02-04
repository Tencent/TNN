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

#include "tnn/device/cpu/acc/cpu_unary_layer_acc.h"
#include "tnn/interpreter/layer_param.h"

#include <cmath>

namespace TNN_NS {

typedef struct hardsigmoid_operator : unary_operator {
public:
    Status Init(LayerParam *param) {
        auto layer_param = dynamic_cast<HardSigmoidLayerParam *>(param);
        if (!layer_param) {
            LOGE("Error: hardsigmoid layer param is nil\n");
            return Status(TNNERR_MODEL_ERR, "Error: hardsigmoid layer param is nil");
        }
        alpha_ = layer_param->alpha;
        beta_  = layer_param->beta;
        min_   = -beta_ / alpha_;
        max_   = (1.0f - beta_) / alpha_;
        return TNN_OK;
    }
    virtual float operator()(float in) {
        float temp = in;
        if (temp <= min_) {
            temp = 0.0;
        } else if (temp < max_) {
            temp = temp * alpha_ + beta_;
        } else {
            temp = 1.0;
        }
        return temp;
    }

private:
    float min_ = 0.f, max_ = 0.f, alpha_ = 0.f, beta_ = 0.f;
} HARDSIGMOID_OP;

DECLARE_UNARY_ACC(HardSigmoid, LAYER_HARDSIGMOID, HARDSIGMOID_OP);

REGISTER_CPU_ACC(HardSigmoid, LAYER_HARDSIGMOID);

}  // namespace TNN_NS
