// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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


DECLARE_OP_CONVERTER(CumSum);

string OnnxOpConverterCumSum::TNNOpType(NodeProto& node,
                                        OnnxNetInfo &net_info) {
    return "Cumsum";
}

string OnnxOpConverterCumSum::TNNLayerParam(NodeProto& node,
                                            OnnxNetInfo& net_info) {
    const std::string& onnx_op = node.op_type();
    ostringstream layer_param;

    std::vector<int64_t> axis_vec = get_node_attr_ai(node, "axis", net_info, 1);
    if (axis_vec.size() != 1) {
        DLog("Cumsum axis size != -1, not supported right now.\n");
        assert(0);
    }

    int64_t exclusive        = get_node_attr_i(node, "exclusive", 0);
    int64_t exclusive_extend = 0;   // By ONNX 1.15.0, Cumsum does not support PyTorch 'extend' mode, set to 0 by default.
    int64_t reverse          = get_node_attr_i(node, "reverse", 0);

    layer_param << axis_vec[0] << " ";
    layer_param << exclusive << " ";
    layer_param << exclusive_extend << " ";
    layer_param << reverse << " ";

    return layer_param.str();
}

bool OnnxOpConverterCumSum::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterCumSum::WriteTNNModel(Serializer* net_writer, NodeProto& node, OnnxNetInfo& net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(CumSum, CumSum);
