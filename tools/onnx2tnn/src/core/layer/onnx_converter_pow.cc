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

DECLARE_OP_CONVERTER(Power);

string OnnxOpConverterPower::TNNOpType(NodeProto &node,
                                          OnnxNetInfo &net_info) {
    return "Power";
}

string OnnxOpConverterPower::TNNLayerParam(NodeProto &node,
                                                     OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;
    
    float scale = 1.0;
    float shift = 0.0;
    float exponent = get_node_attr_f(node, "exponent", net_info, 1, 0.0);

//    std::vector<int64_t> pads = get_node_attr_ai(node, "pads", net_info, 1);
//    float value = get_node_attr_f(node, "value", net_info, 2,0.f);
    layer_param <<exponent<<" "<<scale<<" "<<shift<<" ";

    return layer_param.str();
}

bool OnnxOpConverterPower::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterPower::WriteTNNModel(Serializer *net_writer,
                                                  NodeProto &node,
                                                  OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Power, Pow);
