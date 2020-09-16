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

#include <tnn/core/status.h>

#include "tnn/device/rknpu/convert/rknpu_base_layer.h"
#include "tnn/device/rknpu/convert/rknpu_utils.h"

namespace TNN_NS {

DECLARE_RKNPU_LAYER_WEIGHT(Mul, LAYER_MUL)

Status RknpuMulLayer::Convert() {
    auto param    = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    auto resource = dynamic_cast<EltwiseLayerResource *>(resource_);
    CHECK_PARAM_NULL(param);

    int input_size = input_ops_.size();
    if (!((input_size == 1 && resource) || input_size == 2)) {
        return Status(TNNERR_MODEL_ERR, "Error: the Multiply layer input number is not correct");
    }

    std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;

    // input
    if (input_size == 2) {
        inputs.push_back(input_ops_[0]);
        inputs.push_back(input_ops_[1]);
    } else {
        std::vector<int> weight_shape = resource->element_shape;
        std::vector<int> input_shape;
        for (auto dim : input_ops_[0]->GetDims()) {
            input_shape.push_back((int)dim);
        }
        Status calculate_ret = RknpuUtils::CalculateBroadcastSize(weight_shape, resource, input_shape);
        if (calculate_ret != TNN_OK) {
            return calculate_ret;
        }

        std::vector<int> weight_shape_op = {weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]};
        auto weight_const                = RknpuUtils::CreateRknnTensor(
            graph_, layer_name_ + "_weight", weight_shape, resource->element_handle.force_to<void *>(),
            rk::nn::TensorRole::CONST, resource->element_handle.GetDataType());

        inputs.push_back(weight_const);
        inputs.push_back(input_ops_[0]);
    }

    // output
    std::vector<std::vector<int>> output_shapes;
    Status ret = CalculateOutputShape(output_shapes);
    if (ret != TNN_OK)
        return ret;
    auto rk_output =
        RknpuUtils::CreateRknnTensor(graph_, outputs_name_[0], output_shapes[0], NULL, rk::nn::TensorRole::VAR);
    output_ops_.push_back(rk_output);

    graph_->AddOperator(rk::nn::OperatorType::MULTIPLY, inputs, output_ops_, NULL);

    return ret;
}

REGISTER_RKNPU_LAYER(Mul, LAYER_MUL)

}  // namespace TNN_NS
