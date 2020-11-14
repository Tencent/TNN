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

#include "onnx_converter_reduce.h"

string OnnxConverterReduce::TNNLayerParam(NodeProto &node,
                                          OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    std::vector<int64_t> axes = get_node_attr_ai(node, "axes");
    int64_t keepdims          = get_node_attr_i(node, "keepdims");
    layer_param << keepdims << " ";

    for (int64_t axis : axes) {
        layer_param << axis << " ";
    }
    return layer_param.str();
}

REGISTER_OP_CONVERTER_REDUCE(ReduceL1, ReduceL1);

REGISTER_OP_CONVERTER_REDUCE(ReduceL2, ReduceL2);

REGISTER_OP_CONVERTER_REDUCE(ReduceLogSum, ReduceLogSum);

REGISTER_OP_CONVERTER_REDUCE(ReduceLogSumExp, ReduceLogSumExp);

REGISTER_OP_CONVERTER_REDUCE(ReduceMax, ReduceMax);

REGISTER_OP_CONVERTER_REDUCE(ReduceMean, ReduceMean);

REGISTER_OP_CONVERTER_REDUCE(ReduceMin, ReduceMin);

REGISTER_OP_CONVERTER_REDUCE(ReduceProd, ReduceProd);

REGISTER_OP_CONVERTER_REDUCE(ReduceSum, ReduceSum);

REGISTER_OP_CONVERTER_REDUCE(ReduceSumSquare, ReduceSumSquare);
