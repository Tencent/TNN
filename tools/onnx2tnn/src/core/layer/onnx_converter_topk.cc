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

DECLARE_OP_CONVERTER(TopK);

string OnnxOpConverterTopK::TNNOpType(NodeProto &node,
                                           OnnxNetInfo &net_info) {
    return "TopK";
}

string OnnxOpConverterTopK::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    int axis = -1;
    int largest = 1;
    int sorted = 1;

    if (node_has_attr(node, "axis")) {
        axis = get_node_attr_i(node, "axis");
    }
    if (node_has_attr(node, "largest")) {
        largest = get_node_attr_i(node, "largest");
    }
    if (node_has_attr(node, "sorted")) {
        sorted = get_node_attr_i(node, "sorted");
    }
    int k = get_node_attr_i(node, "k");

    layer_param << axis << " " << largest << " " << sorted << " " << k << " ";

    return layer_param.str();
}

bool OnnxOpConverterTopK::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterTopK::WriteTNNModel(Serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    return 0;
}

REGISTER_OP_CONVERTER(TopK, TopK);
