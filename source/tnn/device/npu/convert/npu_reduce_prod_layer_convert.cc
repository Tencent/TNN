//
// Created by 李烨 on 20/7/20.
//
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
#include "graph/attr_value.h"
#include "graph/op/nn_defs.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(ReduceProd, LAYER_REDUCE_PROD)

Status NpuReduceProdLayer::Convert() {
    // parameter and weight of the pooling layer
    auto param = dynamic_cast<ReduceLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    std::vector<int> axes_32         = param->axis;
    std::vector<int> input_shape_vec = input_ops_[0]->GetShape();

    // check if all reduce
    if (param->all_reduce) {
        axes_32.clear();
        for (int i = 0; i < input_shape_vec.size(); i++) {
            axes_32.push_back(i);
        }
    } else {
        for (int i = 0; i < axes_32.size(); i++) {
            if (axes_32[i] < 0) {
                axes_32[i] = input_shape_vec.size() + axes_32[i];
            }
        }
    }

    std::vector<int64_t> axes(axes_32.begin(), axes_32.end());

    auto output = std::make_shared<ge::op::ReduceProd>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_attr_axes(axes);
    output->set_attr_keep_dims(param->keep_dims);
    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(ReduceProd, LAYER_REDUCE_PROD)

}  // namespace TNN_NS