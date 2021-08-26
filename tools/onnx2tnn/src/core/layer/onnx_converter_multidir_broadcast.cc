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

#include "onnx_converter_multidir_broadcast.h"

#include "onnx_utility.h"

std::tuple<int, std::string> OnnxOpConverterMultiBrodcast::GetWeightInputIndexName(NodeProto &node,
                                                                                   OnnxNetInfo &net_info) {
    int weight_input_index = -1;
    int weight_input_count = 0;
    string weight_name     = "";

    std::map<std::string, onnx::TensorProto>::iterator it;
    for (int j = 0; j < node.input_size(); j++) {
        const std::string &input_name = node.input(j);
        it                            = net_info.weights_map.find(input_name);
        if (it != net_info.weights_map.end()) {
            if (weight_input_count <= 0) {
                weight_input_index = j;
                weight_name = input_name;
            } else {
                weight_input_index = -1;
                weight_name     = "";
            }
            weight_input_count++;
        }
    }

    return std::make_tuple(weight_input_index, weight_name);
}

string OnnxOpConverterMultiBrodcast::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    auto weight_input       = GetWeightInputIndexName(node, net_info);
    auto weight_input_index = get<0>(weight_input);
    if (weight_input_index >= 0) {
        ostringstream layer_param;
        layer_param << weight_input_index << " ";
        return layer_param.str();
    }
    return "";
}

bool OnnxOpConverterMultiBrodcast::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    const std::string &onnx_op        = node.op_type();
    std::string name                  = !node.name().empty() ? node.name() : node.output(0);
    const std::string &tnn_layer_type = TNNOpType(node, net_info);

    auto weight_input       = GetWeightInputIndexName(node, net_info);
    auto weight_input_index = get<0>(weight_input);
    auto weight_name        = get<1>(weight_input);
    if (weight_input_index < 0) {
        return false;
    }
    return true;
}

int OnnxOpConverterMultiBrodcast::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    const std::string &onnx_op        = node.op_type();
    std::string name                  = !node.name().empty() ? node.name() : node.output(0);
    const std::string &tnn_layer_type = TNNOpType(node, net_info);

    auto weight_input       = GetWeightInputIndexName(node, net_info);
    auto weight_input_index = get<0>(weight_input);
    auto weight_name        = get<1>(weight_input);
    if (weight_input_index < 0) {
        return 0;
    }

    //写头信息
    net_writer->PutInt(0);  //触发type from string
    net_writer->PutString(tnn_layer_type);
    net_writer->PutString(name);

    //写数据
    const onnx::TensorProto &weight = net_info.weights_map[weight_name];
//    temporarily comment the following code
//    if (weight.data_type() == TensorProto_DataType_FLOAT || weight.data_type() == TensorProto_DataType_DOUBLE) {
//        WriteTensorData(weight, net_writer, net_info.data_type);
//    } else {
//        WriteTensorData(weight, net_writer, DATA_TYPE_AUTO);
//    }
    WriteTensorData(weight, net_writer, DATA_TYPE_AUTO);
    //有权值写入的返回1， 没有的返回0
    return 1;
}
