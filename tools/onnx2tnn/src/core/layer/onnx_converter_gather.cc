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
    const std::string &onnx_op = node.op_type();
    auto tnn_op_type           = TNNOpType(node, net_info);

    int axis = 0;
    if (node_has_attr(node, "axis")) {
        axis = (int)get_node_attr_i(node, "axis");
    }

    //auto indices = get_node_attr_ai(node, "indices", net_info, 1);

    ostringstream layer_param;
    layer_param << axis << " ";
    auto data_iter    = net_info.weights_map.find(node.input(0));
    auto indices_iter = net_info.weights_map.find(node.input(1));
    layer_param << (data_iter == net_info.weights_map.end() ? 0 : 1) << " ";
    layer_param << (indices_iter == net_info.weights_map.end() ? 0 : 1) << " ";

    return layer_param.str();
}

bool OnnxOpConverterGather::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return true;
}

int OnnxOpConverterGather::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    std::string name                  = !node.name().empty() ? node.name() : node.output(0);
    const std::string &tnn_layer_type = TNNOpType(node, net_info);

    //写头信息
    net_writer->PutInt(0);  //触发type from string
    net_writer->PutString(tnn_layer_type);
    net_writer->PutString(name);

    //写数据
    auto data_iter    = net_info.weights_map.find(node.input(0));
    auto indices_iter = net_info.weights_map.find(node.input(1));
    if (data_iter != net_info.weights_map.end()) {
        net_writer->PutInt(1);
        DataType dst_data_type = net_info.data_type;
        if (data_iter->second.data_type() == onnx::TensorProto_DataType_INT32) {
            dst_data_type = DATA_TYPE_INT32;
        }
        WriteTensorData(data_iter->second, net_writer, dst_data_type);
    } else {
        net_writer->PutInt(0);
    }
    if (indices_iter != net_info.weights_map.end()) {
        net_writer->PutInt(1);
        WriteTensorData(indices_iter->second, net_writer, DATA_TYPE_INT32);
    } else {
        net_writer->PutInt(0);
    }
    return 1;
}

REGISTER_OP_CONVERTER(Gather, Gather);
