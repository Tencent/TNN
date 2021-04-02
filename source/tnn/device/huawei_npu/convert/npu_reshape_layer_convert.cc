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

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(Reshape, LAYER_RESHAPE)

Status NpuReshapeLayer::Convert() {
    auto param = dynamic_cast<ReshapeLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    ge::AttrValue::LIST_INT shape = std::vector<int64_t>(param->shape.begin(), param->shape.end());
    if (shape.size() > 4) {
        LOGE("(Reshape) dims size bigger than 4 is not support in HUAWEI_NPU\n");
        return Status(TNNERR_MODEL_ERR, "(Reshape) dims size bigger than 4 is not support in HUAWEI_NPU");
    }

    if (param->reshape_type == 0) {
        // onnx caffe reshape(nchw): 0
        auto output = std::make_shared<ge::op::Reshape>(outputs_name_[0]);
        output->set_input_tensor(*input_ops_[0]->GetOperator());
        output->set_attr_shape(shape);
        output->set_attr_axis(param->axis);
        output->set_attr_num_axes(param->num_axes);
        ADD_OUTPUT_OP(output)
    } else {
        //// Tensorflow TFLite reshape(nhwc): 1
        //// convert input form nchw to nhwc first
        //auto permute_op = std::make_shared<hiai::op::Permute>(layer_name_ + "permute");
        //permute_op->set_input_x(*input_ops_[0]->GetOperator());
        //permute_op->set_attr_order({0, 2, 3, 1});
        //weight_ops_.push_back(permute_op);

        //// do reshape
        //auto reshape_op = std::make_shared<ge::op::Reshape>(layer_name_ + "reshape");
        //reshape_op->set_input_tensor(*permute_op);
        //reshape_op->set_attr_shape(shape);
        //reshape_op->set_attr_axis(param->axis);
        //reshape_op->set_attr_num_axes(param->num_axes);
        //weight_ops_.push_back(reshape_op);

        //// convert input form nhwc to nchw first
        //auto output = std::make_shared<hiai::op::Permute>(outputs_name_[0]);
        //output->set_input_x(*reshape_op);
        //output->set_attr_order({0, 3, 1, 2});

        //ADD_OUTPUT_OP(output)
        LOGE("TFLite type Reshape is not support in HUAWEI_NPU\n");
        return Status(TNNERR_MODEL_ERR, "TFLite type Reshape is not support in HUAWEI_NPU");
    }
}

REGISTER_NPU_LAYER(Reshape, LAYER_RESHAPE)

}  // namespace TNN_NS
