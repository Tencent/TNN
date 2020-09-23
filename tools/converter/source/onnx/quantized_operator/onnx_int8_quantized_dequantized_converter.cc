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

#include "tools/converter/source/onnx/onnx_base_converter.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(QuantizedAndDequantized);

std::string OnnxQuantizedAndDequantizedConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    if (node.op_type() == "Int8Quantize") {
        return "Int8Quantized";
    } else if (node.op_type() == "Int8Dequantize") {
        return "Int8Dequantized";
    }
    return "";
}

TNN_NS::ActivationType OnnxQuantizedAndDequantizedConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxQuantizedAndDequantizedConverter::exec(tnn::NetStructure &net_structure, tnn::NetResource &net_resource,
                                                const onnx::NodeProto &node,
                                                std::map<std::string, const onnx::TensorProto *> proxy_initializers_map,
                                                std::map<std::string, std::shared_ptr<OnnxProxyNode>> proxy_nodes,
                                                bool &quantized_model) {
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(QuantizedAndDequantized, Int8Quantize);
REGISTER_CONVERTER(QuantizedAndDequantized, Int8Dequantize);


}  // namespace TNN_CONVERTER