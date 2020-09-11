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
#include "graph/op/all_ops.h"
#include "graph/op/nn_defs.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(StridedSlice, LAYER_STRIDED_SLICE);

Status NpuStridedSliceLayer::Convert() {
    auto param    = dynamic_cast<StrideSliceLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    std::vector<int> input_shape_vec = input_ops_[0]->GetShape();

    auto begins = param->begins;
    std::reverse(begins.begin(), begins.end());
    auto ends = param->ends;
    std::reverse(ends.begin(), ends.end());
    auto strides = param->strides;
    std::reverse(strides.begin(), strides.end());

    for (int i = 0; i < ends.size(); ++i) {
        if (ends[i] == 0) {
            ends[i] = input_shape_vec[i];
        }
    }

    ge::Shape input_shape({4});
    ge::TensorDesc desc(input_shape, ge::FORMAT_NCHW, ge::DT_INT32);

    // begins
    std::shared_ptr<ge::op::Const> begins_op = std::make_shared<ge::op::Const>(layer_name_ + "_begin");
    NpuUtils::CreateAttrArray(begins_op, begins, desc, 4);
    weight_ops_.push_back(begins_op);

    // ends
    // in format nchw
    std::shared_ptr<ge::op::Const> ends_op = std::make_shared<ge::op::Const>(layer_name_ + "_end");
    NpuUtils::CreateAttrArray(ends_op, ends, desc, 4);
    weight_ops_.push_back(ends_op);

    // strides
    // in format nchw
    std::shared_ptr<ge::op::Const> strides_op = std::make_shared<ge::op::Const>(layer_name_ + "_stride");
    NpuUtils::CreateAttrArray(strides_op, strides, desc, 4);
    weight_ops_.push_back(strides_op);

    // stride
    auto output = std::make_shared<ge::op::StridedSlice>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_input_begin(*begins_op);
    output->set_input_end(*ends_op);
    output->set_input_strides(*strides_op);
    output->set_attr_begin_mask(0);
    output->set_attr_end_mask(0);
    output->set_attr_ellipsis_mask(0);
    output->set_attr_new_axis_mask(0);
    output->set_attr_shrink_axis_mask(0);
    ADD_OUTPUT_OP(output);
}

REGISTER_NPU_LAYER(StridedSlice, LAYER_STRIDED_SLICE);

}  // namespace TNN_NS
