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

DECLARE_OP_CONVERTER(Expand);

string OnnxOpConverterExpand::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "Expand";
}

string OnnxOpConverterExpand::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
//    const std::string &input_name = node.input(0);
//    // skip weight reshape
//    if (net_info.weights_map.find(input_name) != net_info.weights_map.end()) {
//        DLog("expand of weights is not supported, input0:%s out_node:%s\n", node.input(0).c_str(),
//             node.output(0).c_str());
//        assert(0);
//    }

    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;
    const auto &shape_name = node.input(1);
    if (net_info.weights_map.find(shape_name) != net_info.weights_map.end()) {
        const onnx::TensorProto &shape_tp = net_info.weights_map[node.input(1)];
        auto shape_data                   = (const int64_t *)get_tensor_proto_data(shape_tp);
        int shape_dim_size                = get_tensor_proto_data_size(shape_tp);
        layer_param << shape_dim_size << " ";
        for (int i = 0; i < shape_dim_size; ++i) {
            layer_param << shape_data[i] << " ";
        }
    } else {
        int shape_dim_size = 0;
        layer_param << shape_dim_size << " ";
    }
    return layer_param.str();
}

bool OnnxOpConverterExpand::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterExpand::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Expand, Expand);
