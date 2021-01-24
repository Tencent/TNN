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

#include <fstream>
#include <iostream>
#include <sstream>
#include "onnx_op_converter.h"
#include "onnx_utility.h"



DECLARE_OP_CONVERTER(InstNorm);

string OnnxOpConverterInstNorm::TNNOpType(NodeProto& node,
                                               OnnxNetInfo &net_info) {
    return "InstBatchNormCxx";
}

string OnnxOpConverterInstNorm::TNNLayerParam(NodeProto& node,
                                                    OnnxNetInfo& net_info) {
    const onnx::TensorProto& scale = net_info.weights_map[node.input(1)];

    float epsilon = get_node_attr_f(node, "epsilon", 1e-5f);
    int channels = get_tensor_proto_data_size(scale);

    ostringstream layer_param;
    layer_param << epsilon;
    
    return layer_param.str();
}

bool OnnxOpConverterInstNorm::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return true;
}

int OnnxOpConverterInstNorm::WriteTNNModel(Serializer* net_writer,
                                                 NodeProto& node,
                                                 OnnxNetInfo& net_info) {
    const std::string& onnx_op = node.op_type();
    std::string name = !node.name().empty() ? node.name() : node.output(0);
    const std::string& tnn_layer_type = TNNOpType(node,net_info);

    //写头信息
    net_writer->PutInt(0);  //触发type from string
    net_writer->PutString(tnn_layer_type);
    net_writer->PutString(name);

    //写数据
    const onnx::TensorProto& scale = net_info.weights_map[node.input(1)];
    const onnx::TensorProto& b = net_info.weights_map[node.input(2)];

    int channels = get_tensor_proto_data_size(scale);

    const float* scale_data = get_tensor_proto_data(scale);
    const float* b_data = get_tensor_proto_data(b);

    WriteRawData(scale_data, channels, net_writer, net_info.data_type, {channels});
    WriteRawData(b_data, channels, net_writer, net_info.data_type, {channels});


    //有权值写入的返回1， 没有的返回0
    return 1;
}

REGISTER_OP_CONVERTER(InstNorm, InstanceNormalization);
