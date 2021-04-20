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

DECLARE_NPU_LAYER_WEIGHT(StridedSliceV2, LAYER_STRIDED_SLICE_V2);

Status NpuStridedSliceV2Layer::Convert() {
    if (NpuUtils::VersionCompare(npu_version_, "100.500.010.010", VCT_SMALLER)) {
        LOGE("StrideSliceV2 is not support in this rom version (%s)\n", npu_version_.c_str());
        return Status(TNNERR_MODEL_ERR, "StrideSliceV2 is not support in this rom version");
    }

    auto param = dynamic_cast<StrideSliceV2LayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    std::vector<int> input_shape_vec = input_ops_[0]->GetShape();

    auto begins = param->begins;
    std::reverse(begins.begin(), begins.end());
    auto ends = param->ends;
    std::reverse(ends.begin(), ends.end());
    auto axes = param->axes;
    std::reverse(axes.begin(), axes.end());
    auto strides = param->strides;
    std::reverse(strides.begin(), strides.end());

    for (int i = 0; i < ends.size(); ++i) {
        if (ends[i] == INT_MAX) {
            ends[i] = input_shape_vec[axes[i]];
        } else if (ends[i] == INT_MIN) {
            ends[i] = -1;
        } else if (ends[i] < 0) {
            ends[i] += input_shape_vec[axes[i]];
        }
    }

    ge::Shape param_shape({(int64_t)axes.size()});
    ge::TensorDesc desc(param_shape, ge::FORMAT_NCHW, ge::DT_INT32);

    // begins
    std::shared_ptr<ge::op::Const> begins_op = std::make_shared<ge::op::Const>(layer_name_ + "_begin");
    NpuUtils::CreateAttrArray(begins_op, begins, desc, begins.size());
    weight_ops_.push_back(begins_op);

    // ends
    // in format nchw
    std::shared_ptr<ge::op::Const> ends_op = std::make_shared<ge::op::Const>(layer_name_ + "_end");
    NpuUtils::CreateAttrArray(ends_op, ends, desc, ends.size());
    weight_ops_.push_back(ends_op);

    // axes
    // in format nchw
    std::shared_ptr<ge::op::Const> axes_op = std::make_shared<ge::op::Const>(layer_name_ + "_axes");
    NpuUtils::CreateAttrArray(axes_op, axes, desc, axes.size());
    weight_ops_.push_back(axes_op);

    // strides
    // in format nchw
    std::shared_ptr<ge::op::Const> strides_op = std::make_shared<ge::op::Const>(layer_name_ + "_stride");
    NpuUtils::CreateAttrArray(strides_op, strides, desc, strides.size());
    weight_ops_.push_back(strides_op);

    // stride
    auto output = std::make_shared<hiai::op::StridedSliceV2>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_input_begin(*begins_op);
    output->set_input_end(*ends_op);
    output->set_input_axes(*axes_op);
    output->set_input_strides(*strides_op);
    output->set_attr_begin_mask(0);
    output->set_attr_end_mask(0);
    output->set_attr_ellipsis_mask(0);
    output->set_attr_new_axis_mask(0);
    output->set_attr_shrink_axis_mask(0);
    ADD_OUTPUT_OP(output);
}

REGISTER_NPU_LAYER(StridedSliceV2, LAYER_STRIDED_SLICE_V2);

}  // namespace TNN_NS
