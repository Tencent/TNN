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

#include "graph/attr_value.h"
#include "graph/op/math_defs.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER(Upsample, LAYER_UPSAMPLE)

Status NpuUpsampleLayer::Convert() {
    auto param = dynamic_cast<UpsampleLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    const int scale_h = param->scales[1];
    const int scale_w = param->scales[0];

    auto output = std::make_shared<hiai::op::Upsample>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_attr_stride_h(scale_h);
    output->set_attr_stride_w(scale_w);
    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(Upsample, LAYER_UPSAMPLE)

}  // namespace TNN_NS
