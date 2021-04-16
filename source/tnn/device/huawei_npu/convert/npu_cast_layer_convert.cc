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
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER(Cast, LAYER_CAST)

Status NpuCastLayer::Convert() {
    auto param = dynamic_cast<CastLayerParam*>(param_);
    CHECK_PARAM_NULL(param);

    if (param->from == param->to) {
        auto output = std::make_shared<hiai::op::Permute>(outputs_name_[0]);

        std::vector<int64_t> order;
        order.clear();
        for (int idx = 0; idx < output_shapes_[0].size(); ++idx) {
            order.push_back(idx);
        }

        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_attr_order(order);

        ADD_OUTPUT_OP(output)
    } else {
        auto output = std::make_shared<hiai::op::CastT>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        ge::DataType src_dtype = NpuUtils::ConvertToHiaiDataType((TNN_NS::DataType)param->from);
        ge::DataType dst_dtype = NpuUtils::ConvertToHiaiDataType((TNN_NS::DataType)param->to);
        output->set_attr_src_dtype(src_dtype);
        output->set_attr_src_dtype(dst_dtype);
        ADD_OUTPUT_OP(output)
    }
}

REGISTER_NPU_LAYER(Cast, LAYER_CAST)

}  // namespace TNN_NS
