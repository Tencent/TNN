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

#include <math.h>

namespace TNN_NS {

typedef struct selu_operator : unary_operator {
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
    virtual float operator()(float in) {
        float temp = in;
        if (temp <= 0) {
            temp = gamma_ * (alpha_ * exp(temp) - alpha_);
        } else {
            temp = gamma_ * temp;
        }
        return temp;
    }

private:
    float alpha_ = 0.f, gamma_ = 0.f;
} SELU_OP;

DECLARE_UNARY_ACC(Selu, LAYER_SELU, SELU_OP);

REGISTER_CPU_ACC(Selu, LAYER_SELU);

}  // namespace TNN_NS
