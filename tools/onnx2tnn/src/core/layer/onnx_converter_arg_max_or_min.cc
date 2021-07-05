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

DECLARE_OP_CONVERTER(ArgMaxOrMin);

string OnnxOpConverterArgMaxOrMin::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "ArgMaxOrMin";
}

string OnnxOpConverterArgMaxOrMin::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    int64_t axis              = get_node_attr_i(node, "axis", 0);
    int64_t keepdims          = get_node_attr_i(node, "keepdims", 1);
    int64_t select_last_index = 0;
    if (net_info.opset >= 12) {
        select_last_index = get_node_attr_i(node, "select_last_index", 0);
    }
    if (select_last_index != 0) {
        DLog("ArgMaxOrMin: do not support select last index for now.\n");
        assert(0);
    }
    if (onnx_op == "ArgMin"){
        layer_param << 0 << " ";
    } else if (onnx_op == "ArgMax") {
        layer_param << 1 << " ";
    } else {
        DLog("ArgMaxOrMin: do not support type.\n");
        assert(0);
    }
    layer_param << axis << " ";
    layer_param << keepdims << " ";
    layer_param << select_last_index << " ";

    return layer_param.str();
}

bool OnnxOpConverterArgMaxOrMin::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterArgMaxOrMin::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    return 0;
}

REGISTER_OP_CONVERTER(ArgMaxOrMin, ArgMax);
REGISTER_OP_CONVERTER(ArgMaxOrMin, ArgMin);
