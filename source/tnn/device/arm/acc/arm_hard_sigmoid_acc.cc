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

#include "tnn/device/arm/acc/arm_unary_layer_acc.h"

namespace TNN_NS {

typedef struct arm_hard_sigmoid_operator : arm_unary_operator {
public:
    Status Init(LayerParam *param) {
        auto layer_param = dynamic_cast<HardSigmoidLayerParam *>(param);
        if (!layer_param) {
            LOGE("Error: layer param is nil\n");
            return Status(TNNERR_MODEL_ERR, "Error:  layer param is nil");
        }
        alpha_ = layer_param->alpha;
        beta_  = layer_param->beta;
        minV_  = -beta_ / alpha_;
        maxV_  = (1.0f - beta_) / alpha_;
        return TNN_OK;
    }
    virtual Float4 operator()(const Float4 &v) {
        Float4 val = v;
        Float4 res = Float4::bsl_cle(val, Float4(minV_), Float4(0.f), val * alpha_ + Float4(beta_));
        return Float4::bsl_cge(val, Float4(maxV_), Float4(1.f), res);
    }

private:
    float alpha_ = 1.f;
    float beta_  = 1.f;
    float minV_  = 1.f;
    float maxV_  = 1.f;
} ARM_HARD_SIGMOID_OP;

DECLARE_ARM_UNARY_ACC(HardSigmoid, ARM_HARD_SIGMOID_OP);

REGISTER_ARM_ACC(HardSigmoid, LAYER_HARDSIGMOID);
REGISTER_ARM_LAYOUT(LAYER_HARDSIGMOID, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
