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

#include <graph/op/all_ops.h>
#include "tnn/device/huawei_npu/convert/npu_base_layer_convert.h"

namespace TNN_NS {

DECLARE_NPU_LAYER(Pow, LAYER_POWER)

Status NpuPowLayer::Convert() {
    auto param = dynamic_cast<PowLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    auto output = std::make_shared<hiai::op::Power>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_attr_scale(param->scale);
    output->set_attr_shift(param->shift);
    output->set_attr_power(param->exponent);

    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(Pow, LAYER_POWER);

}  // namespace TNN_NS
