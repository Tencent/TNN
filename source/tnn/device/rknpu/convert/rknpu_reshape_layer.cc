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

#include "rknpu_base_layer.h"
#include "rknpu_utils.h"

namespace TNN_NS {

DECLARE_RKNPU_LAYER(Reshape, LAYER_RESHAPE)

void AddPermute(rk::nn::Graph *graph, std::shared_ptr<rk::nn::Tensor> input, std::shared_ptr<rk::nn::Tensor> output,
                std::vector<uint32_t> perm) {
    std::vector<std::shared_ptr<rk::nn::Tensor>> inputs, outputs;
    inputs.push_back(input);
    outputs.push_back(output);

    rk::nn::PermuteAttr attr;
    attr.perm.push_back(perm[0]);
    attr.perm.push_back(perm[1]);
    attr.perm.push_back(perm[2]);
    attr.perm.push_back(perm[3]);
    graph->AddOperator(rk::nn::OperatorType::PERMUTE, inputs, outputs, (void *)&attr);
}

Status RknpuReshapeLayer::Convert() {
    auto param = dynamic_cast<ReshapeLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    Status ret = TNN_OK;
    std::vector<std::shared_ptr<rk::nn::Tensor>> inputs, outputs;

    if (param->reshape_type == 0) {
        // input
        inputs.push_back(input_ops_[0]);

        // output
        ADD_OUTPUT_OP();

        // ???
        // param->axis
        // param->num_axes

        rk::nn::ReshapeAttr attr;
        for (const auto dim : output_shapes[0]) {
            attr.shapes.push_back(static_cast<uint32_t>(dim));
        }
        graph_->AddOperator(rk::nn::OperatorType::RESHAPE, inputs, output_ops_, (void *)&attr);

    } else if (param->reshape_type == 1) {  // tensorflow reshape, need NCHW => NHWC => Reshape => NCHW
        // output nchw
        ADD_OUTPUT_OP();

        // output nhwc
        auto dims_in               = input_ops_[0]->GetDims();
        std::vector<int> dims_nhwc = {(int)dims_in[0], (int)dims_in[2], (int)dims_in[3], (int)dims_in[1]};
        auto output_nhwc =
            RknpuUtils::CreateRknnTensor(graph_, outputs_name_[0] + "_nhwc", dims_nhwc, NULL, rk::nn::TensorRole::VAR);

        // nchw => nhwc
        AddPermute(graph_, input_ops_[0], output_nhwc, std::vector<uint32_t>{0, 2, 3, 1});

        // input nhwc
        inputs.push_back(output_nhwc);

        // output reshape nhwc
        auto dims_out                 = output_shapes[0];
        std::vector<int> dims_reshape = {dims_out[0], dims_out[2], dims_out[3], dims_out[1]};
        auto output_reshape = RknpuUtils::CreateRknnTensor(graph_, outputs_name_[0] + "_reshape", dims_reshape, NULL,
                                                           rk::nn::TensorRole::VAR);
        outputs.push_back(output_reshape);
        rk::nn::ReshapeAttr attr;
        for (const auto dim : dims_reshape) {
            attr.shapes.push_back(static_cast<uint32_t>(dim));
        }
        graph_->AddOperator(rk::nn::OperatorType::RESHAPE, inputs, outputs, (void *)&attr);

        // nhwc => nchw
        AddPermute(graph_, output_reshape, output_ops_[0], std::vector<uint32_t>{0, 3, 1, 2});

    } else {
        return Status(TNNERR_PARAM_ERR, "Error: ReshapeLayer dont support reshape_type");
    }

    return ret;
}

REGISTER_RKNPU_LAYER(Reshape, LAYER_RESHAPE)

}  // namespace TNN_NS
