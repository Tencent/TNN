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

#include "rknpu_base_layer.h"
#include "rknpu_utils.h"

namespace TNN_NS {

DECLARE_RKNPU_LAYER(ReduceMean, LAYER_REDUCE_MEAN)

Status RknpuReduceMeanLayer::Convert() {
    // parameter and weight of the pooling layer
    auto param = dynamic_cast<ReduceLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    Status ret = TNN_OK;
    std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;

    // input
    inputs.push_back(input_ops_[0]);

    // output
    ADD_OUTPUT_OP();

    std::vector<int> axes                 = param->axis;
    std::vector<uint32_t> input_shape_vec = input_ops_[0]->GetDims();

    // check if all reduce
    if (param->all_reduce) {
        axes.clear();
        for (int i = 0; i < input_shape_vec.size(); i++) {
            axes.push_back(i);
        }
    } else {
        for (int i = 0; i < axes.size(); i++) {
            if (axes[i] < 0) {
                axes[i] = input_shape_vec.size() + axes[i];
            }
        }
    }

    rk::nn::ReduceAttr attr;
    attr.type = rk::nn::ReduceType::REDUCE_MEAN;
    for (const auto val : axes) {
        attr.axis.push_back(static_cast<uint32_t>(val));
    }
    attr.axis_num = attr.axis.size();
    attr.keep_dim = param->keep_dims;
    graph_->AddOperator(rk::nn::OperatorType::REDUCE, inputs, output_ops_, (void *)&attr);

    return ret;
}

REGISTER_RKNPU_LAYER(ReduceMean, LAYER_REDUCE_MEAN)

}  // namespace TNN_NS