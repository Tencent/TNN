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

#include <sstream>

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(MatMul, LAYER_MATMUL)

Status NpuMatMulLayer::Convert() {
    auto param = dynamic_cast<MatMulLayerParam*>(param_);
    CHECK_PARAM_NULL(param);
    auto resource = dynamic_cast<MatMulLayerResource*>(resource_);
    if ((param->weight_position != 0 && param->weight_position != 1) && !resource) {
        return Status(TNNERR_MODEL_ERR, "MatMul resource is invalid");
    }

    auto output = std::make_shared<hiai::op::MatMul>(outputs_name_[0]);

    std::stringstream input0_dims_stream;
    auto input0_dims = input_ops_[0]->GetShape();
    std::copy(input0_dims.begin(), input0_dims.end(), std::ostream_iterator<int>(input0_dims_stream, ","));
    if (input_ops_.size() == 1) {
        auto weight_dims = resource->weight.GetBufferDims();
        if (input0_dims.size() != 2 || weight_dims.size() != 2) {
            std::stringstream weight_dims_stream;
            std::copy(weight_dims.begin(), weight_dims.end(), std::ostream_iterator<int>(weight_dims_stream, ","));
            LOGE("the inputs of MatMul is not 2-dimensional (input1: %s  input2: %s)", input0_dims_stream.str().c_str(),
                 weight_dims_stream.str().c_str());
            return Status(TNNERR_MODEL_ERR, "MatMul in HUAWEI_NPU just support 2-dimensional for both inputs");
        }

        std::shared_ptr<ge::op::Const> data_const = std::make_shared<ge::op::Const>(layer_name_ + "_data");
        ge::Shape data_shape(NpuUtils::Int32VecToTVec<int64_t>(weight_dims));
        NpuUtils::CreateAttrValue(data_const, data_shape, resource->weight);
        weight_ops_.push_back(data_const);

        if (param->weight_position == 0) {
            output->set_input_x1(*data_const);
            output->set_input_x2(*input_ops_[0]->GetOperator());
        } else if (param->weight_position == 1) {
            output->set_input_x1(*input_ops_[0]->GetOperator());
            output->set_input_x2(*data_const);
        } else {
            LOGE("weight_position should be 0 or 1\n");
            return Status(TNNERR_MODEL_ERR, "invalid param in MatMul (weight_position should be 0 or 1)");
        }
    } else if (input_ops_.size() == 2) {
        auto input1_dims = input_ops_[0]->GetShape();
        if (input0_dims.size() != 2 || input1_dims.size() != 2) {
            std::stringstream input1_dims_stream;
            std::copy(input1_dims.begin(), input1_dims.end(), std::ostream_iterator<int>(input1_dims_stream, ","));
            LOGE("the inputs of MatMul is not 2-dimensional (input1: %s  input2: %s)", input0_dims_stream.str().c_str(),
                 input1_dims_stream.str().c_str());
            return Status(TNNERR_MODEL_ERR, "MatMul in HUAWEI_NPU just support 2-dimensional for both inputs");
        }

        output->set_input_x1(*input_ops_[0]->GetOperator());
        output->set_input_x2(*input_ops_[1]->GetOperator());
    } else {
        return Status(TNNERR_MODEL_ERR, "invalid input count in MatMul");
    }

    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(MatMul, LAYER_MATMUL)

}  // namespace TNN_NS
