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

DECLARE_OP_CONVERTER_WITH_FUNC(Reshape,
                               virtual std::vector<std::string> GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info););

string OnnxOpConverterReshape::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "Reshape";
}

std::vector<std::string> OnnxOpConverterReshape::GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info) {
    std::vector<std::string> inputs = {node.input(0)};
    if (node.input_size() == 2 && net_info.weights_map.find(node.input(1)) == net_info.weights_map.end()) {
        inputs.push_back(node.input(1));
    }
    return inputs;
}

string OnnxOpConverterReshape::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;
    const auto &weight_name = node.input(1);
    auto iter               = net_info.weights_map.find(weight_name);
    if (iter != net_info.weights_map.end()) {
        const auto &shape_tp = net_info.weights_map[weight_name];
        auto shape_data      = (const int64_t *)get_tensor_proto_data(shape_tp);
        int data_size        = get_tensor_proto_data_size(shape_tp);
        int start_axis       = 0;
//        int end_axis         = data_size < 4 ? 4 : data_size;
//        int shape_size       = data_size < 4 ? 4 : data_size;

        int end_axis = data_size;
        int shape_size = data_size;

        layer_param << start_axis << " ";
        layer_param << end_axis << " ";
        layer_param << shape_size << " ";
        for (int i = 0; i < shape_size; ++i) {
            if (i < data_size) {
                layer_param << shape_data[i] << " ";
            } else {
                layer_param << 1 << " ";
            }
        }
    } else {
        int start_axis = 0;
        int end_axis   = 0;
        int shape_size = 0;
        layer_param << start_axis << " ";
        layer_param << end_axis << " ";
        layer_param << shape_size << " ";
    }
    int reshape_type = 0;
    layer_param << reshape_type << " ";

    return layer_param.str();
}

bool OnnxOpConverterReshape::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterReshape::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Reshape, Reshape);
