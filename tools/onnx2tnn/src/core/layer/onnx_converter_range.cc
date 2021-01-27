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

#include "onnx_op_converter.h"
#include "onnx_utility.h"

DECLARE_OP_CONVERTER_WITH_FUNC(Range,
                               virtual std::vector<std::string> GetInputNames(NodeProto &node, OnnxNetInfo &net_info););

//void OnnxOpConverterRange::ProcessConstantNode(NodeProto &node, OnnxNetInfo &net_info) {
//    for (const auto &input_node_name : node.input()) {
//        if (net_info.const_node_map.find(input_node_name) != net_info.const_node_map.end()) {
//            net_info.used_const_node.insert(input_node_name);
//        }
//    }
//}

string OnnxOpConverterRange::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "Range";
}

std::vector<std::string> OnnxOpConverterRange::GetInputNames(NodeProto &node, OnnxNetInfo &net_info) {
    //start, limit, delta
    return {node.input(0), node.input(1), node.input(2)};
}

string OnnxOpConverterRange::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    return "";
}

bool OnnxOpConverterRange::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterRange::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    return 0;
}

REGISTER_OP_CONVERTER(Range, Range);
