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

typedef struct arm_selu_operator : arm_unary_operator {
public:
    Status Init(LayerParam *param) {
        auto layer_param = dynamic_cast<SeluLayerParam *>(param);
        CHECK_PARAM_NULL(layer_param);
        alpha_ = layer_param->alpha;
        gamma_ = layer_param->gamma;
        return TNN_OK;
    }

    virtual Float4 operator()(const Float4 &v) {
        return Float4::bsl_cle(v, 0.f, (Float4::exp(v) - 1.0f) * alpha_, v) * gamma_;
    }

private:
    float alpha_ = 0.f;
    float gamma_ = 0.f;
} ARM_SELU_OP;

DECLARE_ARM_UNARY_ACC(Selu, ARM_SELU_OP);
REGISTER_ARM_ACC(Selu, LAYER_SELU);
REGISTER_ARM_LAYOUT(LAYER_SELU, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
