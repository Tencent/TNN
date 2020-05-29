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

#include "half_utils.h"

DECLARE_OP_CONVERTER(MatMul);

//note MatMul convert to gemm by fuse
string OnnxOpConverterMatMul::TNNOpType(NodeProto& node, OnnxNetInfo& net_info) {
    return "MatMul";
}

string OnnxOpConverterMatMul::TNNLayerParam(NodeProto& node,
                                               OnnxNetInfo& net_info) {
    ostringstream layer_param;

    return layer_param.str();
}

int OnnxOpConverterMatMul::WriteTNNModel(serializer* net_writer,
                                            NodeProto& node,
                                            OnnxNetInfo& net_info) {
    // matmul of two tensors
    if (node.input_size() == 2) {
        return 0;
    }

    // matmul of a tensor and its weights
    
    const std::string& onnx_op = node.op_type();
    std::string name = !node.name().empty() ? node.name() : node.output(0);
    const std::string& tnn_layer_type = TNNOpType(node, net_info);

    float alpha = 1.0f;
    float beta  = 1.0f;

    int axis = 1;  // Fix TODO

    //写头信息
    net_writer->put_int(0);  //触发type from string
    net_writer->put_string(tnn_layer_type);
    net_writer->put_string(name);

    //写数据
    //对应innerproduct_data的反序列化
    net_writer->put_string(name);

    const onnx::TensorProto& weights =
        net_info.weights_map[node.input(1)];
    WriteTensorData(weights, net_writer, net_info.data_type);

    // write  
    onnx::TensorProto bias;
    bias.add_float_data(0.f);
    bias.set_data_type(1); // float
    WriteTensorData(bias, net_writer, net_info.data_type);

    //有权值写入的返回1， 没有的返回0
    return 1;
}

REGISTER_OP_CONVERTER(MatMul, MatMul);
