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

DECLARE_RKNPU_LAYER(Slice, LAYER_SLICE);

Status RknpuSliceLayer::Convert() {
    auto param = dynamic_cast<SliceLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    Status ret = TNN_OK;
    std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;

    // input
    inputs.push_back(input_ops_[0]);

    // output
    ADD_OUTPUT_OP();

    rk::nn::SliceAttr attr;

    for (const auto dim : output_shapes[0]) {
        attr.start.push_back(0);
        attr.length.push_back(static_cast<uint32_t>(dim));
    }

    // struct SliceLayerParam : public LayerParam {
    //     // size of each slice
    //     std::vector<int> slices;
    //     int axis;
    // };  SliceLayerParam 参数如何转为 starts / ends ?
    //
    // const auto input_dims = shaper_[input];
    // for (size_t i = 0; i < v_axes.size(); i++) {
    //     int32_t dim = input_dims[v_axes[i]];
    //     if (dim > 0) {
    //         int32_t start = v_starts[i] < 0 ? (v_starts[i] + dim) : v_starts[i];
    //         attr.start[v_axes[i]] = std::max(start, 0);
    //     }
    // }

    graph_->AddOperator(rk::nn::OperatorType::SLICE, inputs, output_ops_, (void *)&attr);

    return ret;
}

REGISTER_RKNPU_LAYER(Slice, LAYER_SLICE);

}  // namespace TNN_NS
