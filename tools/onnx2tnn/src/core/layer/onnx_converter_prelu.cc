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

DECLARE_OP_CONVERTER(PRelu);

string OnnxOpConverterPRelu::TNNOpType(NodeProto &node,
                                            OnnxNetInfo &net_info) {
    return "PReLU";
}

string OnnxOpConverterPRelu::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    const onnx::TensorProto &slope = net_info.weights_map[node.input(1)];
    int num_slope                  = get_tensor_proto_data_size(slope);

    if (num_slope == 1) {
        layer_param << "1 0 ";
    } else if (num_slope > 1) {
        layer_param << "0 0 ";
    } else {
        assert(0);
    }
    return layer_param.str();
}

bool OnnxOpConverterPRelu::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return true;
}

int OnnxOpConverterPRelu::WriteTNNModel(Serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    std::string name = !node.name().empty() ? node.name() : node.output(0);
    const std::string &tnn_layer_type = TNNOpType(node, net_info);

    //写头信息
    net_writer->PutInt(0);  //触发type from string
    net_writer->PutString(tnn_layer_type);
    net_writer->PutString(name);

    //写数据
    const onnx::TensorProto &slope = net_info.weights_map[node.input(1)];
    int num_slope                  = get_tensor_proto_data_size(slope);

    net_writer->PutString(name);
    WriteTensorData(slope, net_writer, DATA_TYPE_FLOAT);
    
    //有权值写入的返回1， 没有的返回0
    return 1;
}

REGISTER_OP_CONVERTER(PRelu, PRelu);
