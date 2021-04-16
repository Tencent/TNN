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



DECLARE_OP_CONVERTER_WITH_FUNC(LayerNorm,
                               virtual std::vector<std::string> GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info););

string OnnxOpConverterLayerNorm::TNNOpType(NodeProto& node,
                                               OnnxNetInfo &net_info) {
    return "LayerNorm";
}

std::vector<std::string> OnnxOpConverterLayerNorm::GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info) {
    return GetAllInputNames(node, net_info);
}
string OnnxOpConverterLayerNorm::TNNLayerParam(NodeProto& node,
                                                    OnnxNetInfo& net_info) {
    
    auto axes_size = get_node_attr_i(node, "reduce_axes_size", 0);
    float eps = get_node_attr_f(node, "epsilon", 1e-5f);

    ostringstream layer_param;
    layer_param << axes_size << " " << eps;
    
    return layer_param.str();
}

bool OnnxOpConverterLayerNorm::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterLayerNorm::WriteTNNModel(Serializer* net_writer,
                                                 NodeProto& node,
                                                 OnnxNetInfo& net_info) {
    //similar to LSTM, from 2020.12.20 on, no WriteTNNModel, no LayerResource, use const_resource_map
    return 0;
}

REGISTER_OP_CONVERTER(LayerNorm, LayerNormalization);
