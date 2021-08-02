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

DECLARE_OP_CONVERTER_WITH_FUNC(ScatterElements,
                               virtual std::vector<std::string> GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info););

string OnnxOpConverterScatterElements::TNNOpType(NodeProto& node, OnnxNetInfo& net_info) {
    return "ScatterElements";
}

string OnnxOpConverterScatterElements::TNNLayerParam(NodeProto& node, OnnxNetInfo& net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    int axis = 0;

    if (node_has_attr(node, "axis")) {
        axis = get_node_attr_i(node, "axis");
    }

    layer_param << axis << " ";

    return layer_param.str();
}

std::vector<std::string> OnnxOpConverterScatterElements::GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info) {
    auto iter_data = net_info.weights_map.find(node.input(0));
    if (iter_data == net_info.weights_map.end()) {
        return {node.input(0), node.input(1), node.input(2)};
    } else {
        return {node.input(1), node.input(2)};
    }

}

bool OnnxOpConverterScatterElements::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    auto iter_data = net_info.weights_map.find(node.input(0));
    if (iter_data == net_info.weights_map.end()) {
        return false;
    } else {
        return true;
    }
};

int OnnxOpConverterScatterElements::WriteTNNModel(Serializer* net_writer, NodeProto& node, OnnxNetInfo& net_info) {
    if (!HasLayerResource(node, net_info)) {
        return 0;
    }
    
    const std::string& onnx_op        = node.op_type();
    std::string name                  = !node.name().empty() ? node.name() : node.output(0);
    const std::string& tnn_layer_type = TNNOpType(node, net_info);

    net_writer->PutInt(0);
    net_writer->PutString(tnn_layer_type);
    net_writer->PutString(name);

    const auto& weights_map  = net_info.weights_map;
    const onnx::TensorProto& data = net_info.weights_map[node.input(0)];
    net_writer->PutBool(true);
    WriteTensorData(data, net_writer, DATA_TYPE_FLOAT);

    return 1;
}

REGISTER_OP_CONVERTER(ScatterElements, ScatterElements);
