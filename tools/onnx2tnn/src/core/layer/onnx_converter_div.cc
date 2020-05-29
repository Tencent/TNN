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
#include "onnx_converter_multidir_broadcast.h"
#include "onnx_utility.h"

#include "half_utils.h"

DECLARE_MULTI_BROADCASR_OP_CONVERTER(Div);


string OnnxOpConverterDiv::TNNOpType(NodeProto& node, OnnxNetInfo &net_info) {
    auto weight_input = GetWeightInputIndexName(node, net_info);
    auto weight_input_index = get<0>(weight_input);
    if (weight_input_index == 1) {
        return "Mul";
    }
    return "Div";
}

string OnnxOpConverterDiv::TNNLayerParam(NodeProto& node,
                                                    OnnxNetInfo& net_info) {
    return OnnxOpConverterMultiBrodcast::TNNLayerParam(node, net_info);
}

int OnnxOpConverterDiv::WriteTNNModel(serializer* net_writer,
                                                 NodeProto& node,
                                                 OnnxNetInfo& net_info) {
    const std::string &onnx_op = node.op_type();
    std::string name = !node.name().empty() ? node.name() : node.output(0);
    const std::string &tnn_layer_type = TNNOpType(node, net_info);
    
   auto weight_input = GetWeightInputIndexName(node, net_info);
   auto weight_input_index = get<0>(weight_input);
   auto weight_name = get<1>(weight_input);
    
   if (weight_input_index == 1) {
       //写头信息
       net_writer->put_int(0);  //触发type from string
       net_writer->put_string(tnn_layer_type);
       net_writer->put_string(name);
       
       const onnx::TensorProto &weight = net_info.weights_map[weight_name];
       int size         = get_tensor_proto_data_size(weight);
       const float *mul = get_tensor_proto_data(weight);
       float *div       = new float[size];
       for (int j = 0; j < size; j++) {
           div[j] = 1.0f / mul[j];
       }
       WriteRawData(div, size, net_writer, net_info.data_type);
       delete[] div;
       
       return 1;
   } else {
       return OnnxOpConverterMultiBrodcast::WriteTNNModel(net_writer, node, net_info);
   }
}

REGISTER_MULTI_BROADCASR_OP_CONVERTER(Div, Div);
