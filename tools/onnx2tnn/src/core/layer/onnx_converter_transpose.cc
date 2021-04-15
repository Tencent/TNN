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

DECLARE_OP_CONVERTER(Transpose);

string OnnxOpConverterTranspose::TNNOpType(NodeProto &node,
                                                OnnxNetInfo &net_info) {
//    const std::string &onnx_op = node.op_type();
//
//    std::vector<int64_t> perm = get_node_attr_ai(node, "perm");
//    if (perm.size() > 4) {
//        return "Transpose";
//    } else {
//        return "Transpose3D";
//    }
    return "Permute";
}

string OnnxOpConverterTranspose::TNNLayerParam(NodeProto &node,
                                                OnnxNetInfo &net_info) {
    ostringstream layer_param;

    std::vector<int64_t> perm = get_node_attr_ai(node, "perm");
    layer_param << perm.size() << " ";
    for (int ii = 0; ii < perm.size(); ii++) {
        layer_param << perm[ii] << " ";
    }

    return layer_param.str();
}

bool OnnxOpConverterTranspose::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterTranspose::WriteTNNModel(Serializer *net_writer,
                                             NodeProto &node,
                                             OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Transpose, Transpose);
