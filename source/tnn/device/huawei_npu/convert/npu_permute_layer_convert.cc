
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
#include "graph/op/nn_defs.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER(Permute, LAYER_PERMUTE)

Status NpuPermuteLayer::Convert() {
    auto param = dynamic_cast<PermuteLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    std::vector<int64_t> orders(param->orders.begin(), param->orders.end());

    if (orders.size() > 4) {
        LOGE("(Permute) dims size bigger than 4 is not support in HUAWEI_NPU");
        return Status(TNNERR_MODEL_ERR, "(Permute) dims size bigger than 4 is not support in HUAWEI_NPU");
    }

    if (NpuUtils::VersionCompare(npu_version_, "100.500.010.012", VCT_BIGEQUAL)) {
        auto output = std::make_shared<hiai::op::Permute>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_attr_order(orders);
        ADD_OUTPUT_OP(output)
    } else {
        LOGE("Permute has bug in this rom version and is disabled\n");
        return Status(TNNERR_MODEL_ERR, "Permute has bug in this rom version and is disabled");
    }
}

REGISTER_NPU_LAYER(Permute, LAYER_PERMUTE)

}  // namespace TNN_NS
