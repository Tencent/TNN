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

#include <tnn/utils/data_type_utils.h>
#include "npu_base_layer_convert.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(LRN, LAYER_LRN)

Status NpuLRNLayer::Convert() {
    auto param = dynamic_cast<LRNLayerParam*>(param_);
    CHECK_PARAM_NULL(param);

    float alpha = param->alpha;
    float beta  = param->beta;
    float bias  = param->bias;
    int size    = param->size;

    auto output = std::make_shared<ge::op::LRN>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_attr_lrn_localsize(size);
    output->set_attr_lrn_alpha(alpha);
    output->set_attr_lrn_beta(beta);
    output->set_attr_lrn_k(bias);

    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(LRN, LAYER_LRN)

}  // namespace TNN_NS
