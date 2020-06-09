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

DECLARE_OP_CONVERTER(Gather);

string OnnxOpConverterGather::TNNOpType(NodeProto &node,
                                           OnnxNetInfo &net_info) {
    auto indices = get_node_attr_ai(node, "indices", net_info, 1);
    if (indices.size() == 1) {
        return "StridedSlice";
    }
    return "Gather";
}

string OnnxOpConverterGather::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    auto tnn_op_type = TNNOpType(node, net_info);

    int axis = (int)get_node_attr_i(node, "axis");
    auto indices = get_node_attr_ai(node, "indices", net_info, 1);

    ostringstream layer_param;
    if (tnn_op_type == "StridedSlice") {
        int dimension = 4;
        std::vector<int64_t> all_starts, all_ends, all_steps;
        for (int ii = 0; ii < axis; ii++) {
            all_starts.push_back(0);
            all_ends.push_back(0);
            all_steps.push_back(1);
        }

        all_starts.push_back(indices[0]);
        all_ends.push_back(indices[0] + 1);
        all_steps.push_back(1);

        for (int ii = axis + 1; ii < dimension; ii++) {
            all_starts.push_back(0);
            all_ends.push_back(0);
            all_steps.push_back(1);
        }

        layer_param << all_starts.size() << " ";
        for (int ii = 0; ii < all_starts.size(); ii++) {
            layer_param << all_starts[ii] << " ";
        }
        layer_param << all_ends.size() << " ";
        for (int ii = 0; ii < all_ends.size(); ii++) {
            layer_param << all_ends[ii] << " ";
        }
        layer_param << all_steps.size() << " ";
        for (int ii = 0; ii < all_steps.size(); ii++) {
            layer_param << all_steps[ii] << " ";
        }
    } else {
        layer_param << axis << " ";
        layer_param << indices.size() << " ";
        for (int ii = 0; ii < indices.size(); ii++) {
            layer_param << indices[ii] << " ";
        }
    }

    return layer_param.str();
}

int OnnxOpConverterGather::WriteTNNModel(serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Gather, Gather);
