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

DECLARE_OP_CONVERTER(Split);

string OnnxOpConverterSplit::TNNOpType(NodeProto &node,
                                            OnnxNetInfo &net_info) {
    // caffe split can be removed by caffe2onnx tool
    return "SplitV";
}

string OnnxOpConverterSplit::TNNLayerParam(NodeProto &node,
                                                OnnxNetInfo &net_info) {
    ostringstream layer_param;

    int64_t axis                = get_node_attr_i(node, "axis", 1);

    std::vector<int64_t> splits;
    if (net_info.opset < 13) {
        splits = get_node_attr_ai(node, "split");
    } else {
        if (node.input_size() > 1) {
            const std::string &split_name = node.input(1);
            if (net_info.weights_map.find(split_name) != net_info.weights_map.end()) {
                const auto &split_tensor = net_info.weights_map[split_name];
                auto split_data          = (const int64_t *)get_tensor_proto_data(split_tensor);
                int data_size            = get_tensor_proto_data_size(split_tensor);
                for (int i = 0; i < data_size; ++i) {
                    splits.push_back(split_data[i]);
                }
            }
        }
    }

    layer_param << axis << " " << splits.size() << " ";
    for (int64_t iter : splits) {
        layer_param << iter << " ";
    }

    return layer_param.str();
}

bool OnnxOpConverterSplit::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterSplit::WriteTNNModel(Serializer *net_writer,
                                             NodeProto &node,
                                             OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Split, Split);
