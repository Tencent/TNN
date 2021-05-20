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

DECLARE_NPU_LAYER_WEIGHT(Normalize, LAYER_NORMALIZE)

Status NpuNormalizeLayer::Convert() {
    auto param = dynamic_cast<NormalizeLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    if (param->p == 2) {
        auto output         = std::make_shared<hiai::op::L2Normalize>(outputs_name_[0]);
        int input_dims_size = (int)input_ops_[0]->GetShape().size();
        if (input_dims_size < 4) {
            // reshape dims to dim-4
            auto reshape_op = std::make_shared<hiai::op::Reshape>(layer_name_ + "_reshape");
            std::vector<int> shape;
            shape.clear();
            for (int i = 0; i < 4; ++i) {
                if (i < input_dims_size) {
                    shape.push_back(0);
                } else {
                    shape.push_back(1);
                }
            }
            ge::TensorDesc shape_desc(ge::Shape({(int64_t)shape.size()}), ge::FORMAT_NCHW, ge::DT_INT32);
            std::shared_ptr<ge::op::Const> shape_const =
                std::make_shared<ge::op::Const>(layer_name_ + "_reshape_shape_const");
            RETURN_ON_NEQ(NpuUtils::CreateAttrArray(shape_const, shape, shape_desc, shape.size()), TNN_OK);
            reshape_op->set_input_x(*input_ops_[0]->GetOperator());
            reshape_op->set_input_shape(*shape_const);
            weight_ops_.push_back(shape_const);
            weight_ops_.push_back(reshape_op);

            output->set_input_x(*reshape_op);
        } else if (input_dims_size == 4) {
            output->set_input_x(*input_ops_[0]->GetOperator());
        } else {
            LOGE("input dims (%d) > 4 is not support in Normalize yet\n", input_dims_size);
            return Status(TNNERR_MODEL_ERR, "input dims > 4 is not support in Normalize for HUAWEI_NPU");
        }

        output->set_attr_axis({param->axis});
        if (param->epsilon > 1e-4f)
            output->set_attr_eps(param->epsilon);
        ADD_OUTPUT_OP(output)
    } else {
        LOGE("the param->p (%d) is not support in Normalize yet\n", param->p);
        return Status(TNNERR_MODEL_ERR, "the param p is invalid in Normalize for HUAWEI_NPU");
    }
}

REGISTER_NPU_LAYER(Normalize, LAYER_NORMALIZE)

}  // namespace TNN_NS
