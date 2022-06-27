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

DECLARE_OP_CONVERTER_WITH_FUNC(OneHot, virtual std::vector<std::string> GetValidInputNames(NodeProto &node,
                                                                                           OnnxNetInfo &net_info););

string OnnxOpConverterOneHot::TNNOpType(NodeProto &node,
                                           OnnxNetInfo &net_info) {
    return "OneHot";
}

std::vector<std::string> OnnxOpConverterOneHot::GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info) {
    const int input_size = node.input_size();
    std::vector<std::string> inputs(input_size);
    for (int i = 0; i < input_size; i++) {
        inputs[i] = node.input(i);
    }
    return inputs;
}

string OnnxOpConverterOneHot::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;
    
    int axis = (int)get_node_attr_i(node, "axis", -1);
    
    int depth = -1;
    if (node_has_attr(node, "depth")) {
        depth = (int)get_node_attr_i(node, "depth");
    } else if (node.input_size() >1 && net_info.weights_map.find(node.input(1)) != net_info.weights_map.end()) {
        const onnx::TensorProto& tensorProto = net_info.weights_map.at(node.input(1));
        auto depth_i = get_tensor_proto_data_vector<int>(tensorProto);
        depth = depth_i[0];
    }
    
    layer_param << axis <<" "<< depth << " ";

    return layer_param.str();
}

bool OnnxOpConverterOneHot::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterOneHot::WriteTNNModel(Serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(OneHot, OneHot);
