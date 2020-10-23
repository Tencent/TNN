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

DECLARE_OP_CONVERTER(Gather);

string OnnxOpConverterGather::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "Gather";
}

string OnnxOpConverterGather::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    const auto &weight_map = net_info.weights_map;
    ostringstream layer_param;
    int axis                 = (int)get_node_attr_i(node, "axis", 0);
    bool data_in_resource    = false;
    bool indices_in_resource = false;

    const auto &data_name = node.input(0);
    if (weight_map.find(data_name) != weight_map.end()) {
        data_in_resource = true;
    }
    const auto &indices_name = node.input(1);
    if (weight_map.find(indices_name) != weight_map.end()) {
        indices_in_resource = true;
    }
    layer_param << axis << " ";
    layer_param << (data_in_resource == true ? 1 : 0) << " ";
    layer_param << (indices_in_resource == true ? 1 : 0) << " ";
    return layer_param.str();
}

int OnnxOpConverterGather::WriteTNNModel(serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    const std::string &onnx_op        = node.op_type();
    std::string name                  = !node.name().empty() ? node.name() : node.output(0);
    const std::string &tnn_layer_type = TNNOpType(node, net_info);
    const auto &weight_map            = net_info.weights_map;

    // layer header
    net_writer->put_int(0);
    net_writer->put_string(tnn_layer_type);
    net_writer->put_string(name);

    auto &data_name = node.input(0);
    if (weight_map.find(data_name) != weight_map.end()) {
        net_writer->put_bool(true);
        const auto &data_tensor = weight_map.find(data_name)->second;
        WriteTensorData(data_tensor, net_writer, DATA_TYPE_FLOAT);
    } else {
        net_writer->put_bool(false);
    }
    const auto &indices_name = node.input(1);
    if (weight_map.find(indices_name) != weight_map.end()) {
        net_writer->put_bool(true);
        const auto &indices_tensor = weight_map.find(indices_name)->second;
        WriteTensorData(indices_tensor, net_writer, DATA_TYPE_FLOAT);
    } else {
        net_writer->put_bool(false);
    }
    return 1;
}

REGISTER_OP_CONVERTER(Gather, Gather);
