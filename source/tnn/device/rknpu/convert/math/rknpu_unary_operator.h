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

#ifndef TNN_SOURCE_TNN_DEVICE_RK_NPU_CONVERT_MATH_RKNPU_UNARY_OPERATOR_H_
#define TNN_SOURCE_TNN_DEVICE_RK_NPU_CONVERT_MATH_RKNPU_UNARY_OPERATOR_H_

#include <tnn/core/layer_type.h>
#include <tnn/device/rknpu/convert/rknpu_base_layer.h>
#include <tnn/device/rknpu/convert/rknpu_utils.h>

namespace TNN_NS {

class RknpuUnaryLayer : public RknpuBaseLayer {
public:
    RknpuUnaryLayer(LayerType layer_type) : RknpuBaseLayer(layer_type){};
    virtual ~RknpuUnaryLayer() {}

protected:
    Status UnaryConvert(rk::nn::OperatorType op_type) {
        int input_size = input_ops_.size();
        if (input_size >= 2) {
            return Status(TNNERR_PARAM_ERR, "Error: the Unary layer count is not correct");
        }

        Status ret = TNN_OK;
        std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;

        // input
        inputs.push_back(input_ops_[0]);

        // output
        ADD_OUTPUT_OP();

        graph_->AddOperator(op_type, inputs, output_ops_, NULL);

        return ret;
    }
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_RK_NPU_CONVERT_MATH_RKNPU_UNARY_OPERATOR_H_
