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

DECLARE_OP_CONVERTER_WITH_FUNC(GroupNorm,
                               virtual std::vector<std::string> GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info););

string OnnxOpConverterGroupNorm::TNNOpType(NodeProto& node,
                                               OnnxNetInfo &net_info) {
    return "GroupNorm";
}

std::vector<std::string> OnnxOpConverterGroupNorm::GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info) {
    return GetAllInputNames(node, net_info);
}
string OnnxOpConverterGroupNorm::TNNLayerParam(NodeProto& node,
                                                    OnnxNetInfo& net_info) {
    const onnx::TensorProto& scale = net_info.weights_map[node.input(1)];
    
    auto num_groups = get_node_attr_i(node, "num_groups", 0);
    float eps = get_node_attr_f(node, "eps", 1e-5f);

    ostringstream layer_param;
    layer_param << num_groups << " " << eps;
    
    return layer_param.str();
}

bool OnnxOpConverterGroupNorm::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterGroupNorm::WriteTNNModel(Serializer* net_writer,
                                                 NodeProto& node,
                                                 OnnxNetInfo& net_info) {
    //similar to LSTM, from 2020.12.20 on, no WriteTNNModel, no LayerResource, use const_resource_map
    return 0;
}

REGISTER_OP_CONVERTER(GroupNorm, GroupNormalization);
