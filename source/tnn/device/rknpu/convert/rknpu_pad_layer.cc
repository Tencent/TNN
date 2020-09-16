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

#include <algorithm>

#include "rknpu_base_layer.h"
#include "rknpu_utils.h"

namespace TNN_NS {

DECLARE_RKNPU_LAYER(Pad, LAYER_PAD);

Status RknpuPadLayer::Convert() {
    auto param = dynamic_cast<PadLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    Status ret = TNN_OK;
    std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;

    // input
    inputs.push_back(input_ops_[0]);

    // output
    ADD_OUTPUT_OP();

    rk::nn::PadAttr attr;
    switch (param->type) {
        case 0:
            attr.mode = rk::nn::PadMode::PAD_CONSTANT;
            break;
        case 1:
            attr.mode = rk::nn::PadMode::PAD_REFLECT;
            break;
        case 2:
            attr.mode = rk::nn::PadMode::PAD_REPLICATE;
            break;
        default:
            throw std::invalid_argument("RknpuPadLayer::Convert: unknow pad mode!");
            break;
    }

    int dims_num = input_ops_[0]->GetDims().size();
    int pads_num = param->pads.size() / 2;
    for (int i = 0; i < dims_num; i++) {
        if (i < pads_num) {
            attr.begin.push_back(param->pads[2 * i]);
            attr.end.push_back(param->pads[2 * i + 1]);
        } else {
            attr.begin.push_back(0);
            attr.end.push_back(0);
        }
    }
    std::reverse(attr.begin.begin(), attr.begin.end());
    std::reverse(attr.end.begin(), attr.end.end());
    attr.const_val = 0;

    graph_->AddOperator(rk::nn::OperatorType::PAD, inputs, output_ops_, (void *)&attr);

    return ret;
}

REGISTER_RKNPU_LAYER(Pad, LAYER_PAD);

}  // namespace TNN_NS
