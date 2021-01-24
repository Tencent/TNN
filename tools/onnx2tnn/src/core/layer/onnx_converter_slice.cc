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

DECLARE_OP_CONVERTER(Slice);

string OnnxOpConverterSlice::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "StridedSliceV2";
}

string OnnxOpConverterSlice::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    std::vector<int64_t> starts = get_node_attr_ai(node, "starts", net_info, 1);
    std::vector<int64_t> ends   = get_node_attr_ai(node, "ends", net_info, 2);
    std::vector<int64_t> axes   = get_node_attr_ai(node, "axes", net_info, 3);
    std::vector<int64_t> steps;
    if (net_info.opset >= 10) {
        steps = get_node_attr_ai(node, "steps", net_info, 4);
    }
    layer_param << starts.size() << " ";
    for (const auto &start : starts) {
        layer_param << start << " ";
    }
    layer_param << ends.size() << " ";
    for (const auto &end : ends) {
        if (end == LLONG_MAX) {
            layer_param << INT_MAX << " ";
        } else if (end == LLONG_MAX) {
            layer_param << INT_MIN << " ";
        } else {
            layer_param << end << " ";
        }
    }
    // pad axes size to starts.size
    if (axes.empty()) {
        for (int i = 0; i < starts.size(); ++i) {
            axes.push_back(i);
        }
    }
    layer_param << axes.size() << " ";
    for (const auto &axis : axes) {
        layer_param << axis << " ";
    }
    // Pad steps
    if (steps.empty()) {
        steps = std::vector<int64_t>(starts.size(), 1);
    }
    layer_param << steps.size() << " ";
    for (const auto &step : steps) {
        layer_param << step << " ";
    }
    return layer_param.str();
}

bool OnnxOpConverterSlice::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterSlice::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Slice, Slice);
