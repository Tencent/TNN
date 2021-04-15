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

DECLARE_OP_CONVERTER(BitShift);

string OnnxOpConverterBitShift::TNNOpType(NodeProto &node,
                                           OnnxNetInfo &net_info) {
    return "BitShift";
}

string OnnxOpConverterBitShift::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;
    auto direction_s = get_node_attr_s(node, "direction");
    auto direction = direction_s=="LEFT" ? 1 : 0;
    
    int bits = 0;
    if (node_has_attr(node, "depth")) {
        bits = get_node_attr_i(node, "depth");
    } else if (node.input_size() >1 && net_info.weights_map.find(node.input(1)) != net_info.weights_map.end()) {
        const onnx::TensorProto& tensorProto = net_info.weights_map.at(node.input(1));
        auto depth_i = get_tensor_proto_data_vector<int>(tensorProto);
        bits = depth_i[0];
    }
    
    layer_param << direction << " "<<bits<<" ";

    return layer_param.str();
}

bool OnnxOpConverterBitShift::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterBitShift::WriteTNNModel(Serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(BitShift, BitShift);
