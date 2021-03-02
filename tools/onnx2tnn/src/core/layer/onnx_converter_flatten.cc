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

DECLARE_OP_CONVERTER(Flatten);

string OnnxOpConverterFlatten::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "Flatten";
}

string OnnxOpConverterFlatten::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    if (node.input_size() == 1 || node.input_size() == 2) {
        const std::string &input_name = node.input(0);

        // skip weight Flatten
        if (net_info.weights_map.find(input_name) != net_info.weights_map.end()) {
            DLog("Flatten of weights is not supported, input0:%s out_node:%s\n", node.input(0).c_str(),
                 node.output(0).c_str());
            assert(0);
        }
    }
    ostringstream layer_param;
    int axis = get_node_attr_i(node, "axis");
    layer_param << axis << " ";

    return layer_param.str();
}

bool OnnxOpConverterFlatten::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterFlatten::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Flatten, Flatten);
