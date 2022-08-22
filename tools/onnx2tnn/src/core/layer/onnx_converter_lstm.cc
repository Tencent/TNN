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



DECLARE_OP_CONVERTER_WITH_FUNC(LSTM,
                               std::vector<std::string> GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info););

string OnnxOpConverterLSTM::TNNOpType(NodeProto& node,
                                                OnnxNetInfo &net_info) {
    return "LSTMONNX";
}

std::vector<std::string> OnnxOpConverterLSTM::GetValidInputNames(NodeProto &node, OnnxNetInfo &net_info) {
    std::vector<std::string> input_names;
    for (int j = 0; j < (int)node.input_size(); j++) {
        const auto input_name = node.input(j);
        if (input_name.length() <= 0) {
            continue;
        }
        // skip sequence_lens
        if (j == 4) {
            continue;
        }
        input_names.push_back(input_name);
    }
    return input_names;
}

string OnnxOpConverterLSTM::TNNLayerParam(NodeProto& node,
                                                    OnnxNetInfo& net_info) {
    int hidden_size = (int)get_node_attr_i(node, "hidden_size", 0);
    auto direction_s = get_node_attr_s(node, "direction", "forward");
    int direction = 0;
    if (direction_s == "reverse") {
        direction = 1;
    } else if (direction_s == "bidirectional") {
        direction = 2;
    }

    ostringstream layer_param;
    layer_param <<0<<" "<<hidden_size<<" "<<direction<<" ";
    
    return layer_param.str();
}

bool OnnxOpConverterLSTM::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
};

int OnnxOpConverterLSTM::WriteTNNModel(Serializer* net_writer,
                                                 NodeProto& node,
                                                 OnnxNetInfo& net_info) {
    //write weights in constant resource from now on
    //write weights in constant resource from now on
    //write weights in constant resource from now on
    return 0;
}

REGISTER_OP_CONVERTER(LSTM, LSTM);
