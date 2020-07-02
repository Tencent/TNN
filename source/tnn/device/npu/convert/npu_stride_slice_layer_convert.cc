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
namespace tnn {
DECLARE_NPU_LAYER_WEIGHT(StridedSlice, LAYER_STRIDED_SLICE);
Status NpuStridedSliceLayer::Convert() {
    auto param    = dynamic_cast<StrideSliceLayerParam *>(param_);
    auto resource = dynamic_cast<StrideSliceLayerParam *>(resource_);
    if (!param) {
        LOGE("Error: StrideSlice param is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error:StrideSliceParam is nil");
    }
    auto input_data                  = input_ops_[0];
    std::vector<int> input_shape_vec = input_data->GetShape();

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
    // printf("the begins are %d %d %d %d \n", begins[0], begins[1], begins[2], begins[3]);
    std::shared_ptr<ge::op::Const> begins_op = std::make_shared<ge::op::Const>(layer_name_ + "_begin");
    NpuUtils::CreateAttrArray(begins_op, begins, desc);
    weight_ops_.push_back(begins_op);

    // ends
    // in format nchw
    //  printf("the ends are %d %d %d %d\n", ends[0], ends[1], ends[2], ends[3]);
    std::shared_ptr<ge::op::Const> ends_op = std::make_shared<ge::op::Const>(layer_name_ + "_end");
    NpuUtils::CreateAttrArray(ends_op, ends, desc);
    weight_ops_.push_back(ends_op);

    // strides
    // in format nchw
    // printf("the strides are %d %d %d %d\n", strides[0], strides[1], strides[2], strides[3]);
    std::shared_ptr<ge::op::Const> strides_op = std::make_shared<ge::op::Const>(layer_name_ + "_stride");
    NpuUtils::CreateAttrArray(strides_op, strides, desc);
    weight_ops_.push_back(strides_op);

    // stride
    auto output = std::make_shared<ge::op::StridedSlice>(outputs_[0]);
    output->set_input_x(*input_data->GetOperator());
    output->set_input_begin(*begins_op);
    output->set_input_end(*ends_op);
    output->set_input_strides(*strides_op);
    output->set_attr_begin_mask(0);
    output->set_attr_end_mask(0);
    output->set_attr_ellipsis_mask(0);
    output->set_attr_new_axis_mask(0);
    output->set_attr_shrink_axis_mask(0);
    std::shared_ptr<OperatorInfo> output_op = std::make_shared<OperatorInfo>(output);
    output_ops_.push_back(output_op);
    return SetOutputOps();
}
REGISTER_NPU_LAYER(StridedSlice, LAYER_STRIDED_SLICE);
}  // namespace tnn