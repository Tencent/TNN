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

#include <cmath>
#include <memory>

#include <ngraph/node.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <inference_engine.hpp>

#include "tnn/layer/base_layer.h"
#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/network/openvino/openvino_types.h"

namespace TNN_NS {
namespace openvino {

DECLARE_OPENVINO_LAYER_BUILDER(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

namespace opset = ngraph::opset3;

Status ArgMaxOrMinOVLayerBuilder::Build() {
    if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }

    auto param = dynamic_cast<ArgMaxOrMinLayerParam*>(param_);
    CHECK_PARAM_NULL(param);

    auto topk_mode = param->mode == 0 ? opset::TopK::Mode::MIN : opset::TopK::Mode::MAX ;
    auto input_node = GetInputNodes()[0];
    auto k_node = opset::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
    auto topk = std::make_shared<opset::TopK>(
        input_node, k_node, param->axis, topk_mode, opset::TopK::SortType::NONE);

    std::shared_ptr<ngraph::Node> cur_node;
    if (param->keep_dims == 0) {
        auto axis_to_remove = opset::Constant::create(ngraph::element::i32, ngraph::Shape{}, {param->axis});
        auto reshaped_indices = std::make_shared<opset::Squeeze>(topk->output(1), axis_to_remove);
        cur_node = std::make_shared<opset::Convert>(reshaped_indices, ngraph::element::i64);
    } else {
        cur_node = std::make_shared<opset::Convert>(topk->output(1), ngraph::element::i64);
    }

    cur_node->set_friendly_name(param_->name);
    cur_node->validate_and_infer_types();

    ngraph::NodeVector node_vector = {cur_node};
    SetOutputTensors(node_vector);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

}
}