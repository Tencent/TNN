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
#include "tnn/device/huawei_npu/convert/npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN)

Status NpuArgMaxOrMinLayer::Convert() {
    auto param = dynamic_cast<ArgMaxOrMinLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    if (param->mode == 1) {
        // arg max
        std::shared_ptr<ge::op::Const> axis_const = std::make_shared<ge::op::Const>(layer_name_ + "_axis");
        std::vector<int> axis_vec                 = {param->axis};
        ge::TensorDesc const_desc(ge::Shape({1}), ge::FORMAT_NCHW, ge::DT_INT32);
        NpuUtils::CreateAttrArray(axis_const, axis_vec, const_desc, 1);
        weight_ops_.push_back(axis_const);

        auto output = std::make_shared<hiai::op::ArgMaxExt2>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_input_axis(*axis_const);
        output->set_attr_keep_dims(param->keep_dims);

        ADD_OUTPUT_OP(output)
    } else {
        LOGE("ArgMaxOrMin layer only support max by now\n");
        return Status(TNNERR_PARAM_ERR, "ArgMaxOrMin layer only support max by now\n");
    }
}

REGISTER_NPU_LAYER(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN)

}  // namespace TNN_NS
