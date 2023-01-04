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

#include "onnx/onnx_utils.h"
#include "tnn/interpreter/tnn/objseri.h"
#include "tools/converter/source/onnx/onnx_base_converter.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Slice);

int Int64ToInt32(const int64_t number) {
    if (number < INT_MIN) {
        return INT_MIN;
    } else if (number > INT_MAX) {
        return INT_MAX;
    }

    return (int)number;
}

std::string OnnxSliceConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "StridedSliceV2";
}

TNN_NS::ActivationType OnnxSliceConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxSliceConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                        const onnx::NodeProto &node,
                                        std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                        std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                        bool &quantized_model) {
    const std::string &onnx_op = node.op_type();
    auto param                 = new TNN_NS::StrideSliceV2LayerParam;
    auto cur_layer             = net_structure.layers.back();
    cur_layer->param           = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type                = cur_layer->type_str;
    param->name                = cur_layer->name;
    param->quantized           = false;

    auto starts     = GetAttributeInt64Vector(node, "starts", proxy_initializers_map, 1);
    auto ends       = GetAttributeInt64Vector(node, "ends", proxy_initializers_map, 2);
    auto axes       = GetAttributeInt64Vector(node, "axes", proxy_initializers_map, 3);
    auto steps      = GetAttributeInt64Vector(node, "steps", proxy_initializers_map, 4);

    if (steps.empty()) {
        steps.push_back(1);
    }
    for (int i = 0; i < starts.size(); i++) {
        param->begins.push_back(Int64ToInt32(starts[i]));
    }

    for (int i = 0; i < ends.size(); i++) {
        param->ends.push_back(Int64ToInt32(ends[i]));
    }

    for (int i = 0; i < axes.size(); i++) {
        param->axes.push_back(axes[i]);
    }

    for (int i = 0; i < steps.size(); i++) {
        param->strides.push_back(steps[i]);
    }

    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = node.input(0);

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Slice, Slice);

}  // namespace TNN_CONVERTER
