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



DECLARE_OP_CONVERTER(LSTM);

string OnnxOpConverterLSTM::TNNOpType(NodeProto& node,
                                                OnnxNetInfo &net_info) {
    return "LSTMONNX";
}

string OnnxOpConverterLSTM::TNNLayerParam(NodeProto& node,
                                                    OnnxNetInfo& net_info) {
    if (node.input(4).length() > 0) {
        DLog("Note: sequence_lens  is only supported\n");
        assert(0);
    }
    
    int hidden_size = (int)get_node_attr_i(node, "hidden_size", 0);

    ostringstream layer_param;
    layer_param <<0<<" "<<hidden_size<<" ";
    
    return layer_param.str();
}

bool OnnxOpConverterLSTM::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
};

int OnnxOpConverterLSTM::WriteTNNModel(Serializer* net_writer,
                                                 NodeProto& node,
                                                 OnnxNetInfo& net_info) {
    //write weights in constant resource from now on
//    if (net_info.weights_map.find(node.input(5)) != net_info.weights_map.end() ||
//        net_info.weights_map.find(node.input(6)) != net_info.weights_map.end()) {
//        DLog("Note: Weights of initial_h or initial_c  is only supported by TNN, old Rapidnet dont support\n");
//        assert(0);
//    }
//
//    const auto onnx_op = node.op_type();
//    auto name = !node.name().empty() ? node.name() : node.output(0);
//    const auto tnn_layer_type = TNNOpType(node, net_info);
//
//    //写头信息
//    net_writer->put_int(0);  //触发type from string
//    net_writer->put_string(tnn_layer_type);
//    net_writer->put_string(name);
//
//    //写数据
//    net_writer->put_string(name);
//
//    //write gate weight
//    auto W = get_node_attr_tensor(node, "W", net_info, 1);
//    WriteTensorData(W, net_writer, net_info.data_type);
//
//    //write hidden weight
//    auto R = get_node_attr_tensor(node, "R", net_info, 2);
//    WriteTensorData(R, net_writer, net_info.data_type);
//
//    //write bias
//    auto B = get_node_attr_tensor(node, "B", net_info, 3);
//    WriteTensorData(B, net_writer, net_info.data_type);
    
    return 0;
}

REGISTER_OP_CONVERTER(LSTM, LSTM);
