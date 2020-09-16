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

DECLARE_RKNPU_LAYER_WEIGHT(StridedSlice, LAYER_STRIDED_SLICE);

Status RknpuStridedSliceLayer::Convert() {
    auto param = dynamic_cast<StrideSliceLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    Status ret = TNN_OK;
    std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;

    // input
    inputs.push_back(input_ops_[0]);

    // output
    ADD_OUTPUT_OP();

    rk::nn::StridedSliceAttr attr;

    attr.begin = param->begins;
    std::reverse(attr.begin.begin(), attr.begin.end());
    attr.end = param->ends;
    std::reverse(attr.end.begin(), attr.end.end());
    attr.stride = param->strides;
    std::reverse(attr.stride.begin(), attr.stride.end());

    for (int i = 0; i < attr.end.size(); ++i) {
        if (attr.end[i] == 0) {
            attr.end[i] = input_ops_[0]->GetDims()[i];
        }
    }

    graph_->AddOperator(rk::nn::OperatorType::STRIDED_SLICE, inputs, output_ops_, (void *)&attr);

    return ret;
}

REGISTER_RKNPU_LAYER(StridedSlice, LAYER_STRIDED_SLICE);

}  // namespace TNN_NS
