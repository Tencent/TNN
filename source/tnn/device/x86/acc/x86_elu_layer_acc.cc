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
#include <math.h>

namespace TNN_NS {

typedef struct x86_elu_layer_acc : x86_unary_operator {
public:
    Status Init(LayerParam *param) {
        auto layer_param = dynamic_cast<EluLayerParam *>(param);
        if (!layer_param) {
            LOGE("Error: layer param is nil\n");
            return Status(TNNERR_MODEL_ERR, "Error: layer param is nil");
        }
        alpha_ = layer_param->alpha;
        return TNN_OK;
    }

    virtual float operator()(const float v) {
        float tmp = v;
        if (tmp < 0) {
            tmp = alpha_ * (exp(tmp) - 1.0f);
        }
        return tmp;
    }
private:
    float alpha_ = 0;
} X86_ELU_OP;


DECLARE_X86_UNARY_ACC(Elu, X86_ELU_OP);

REGISTER_X86_ACC(Elu, LAYER_ELU);

}   // namespace TNN_NS
