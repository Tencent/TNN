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

std::vector<int64_t> get_tensor_data(onnx::TensorProto &tensor) {
    const auto data_type = tensor.data_type();
    const auto *data     = get_tensor_proto_data(tensor);
    const int size       = get_tensor_proto_data_size(tensor);
    std::vector<int64_t> data_vec(size, 0);
    for (int i = 0; i < size; i++) {
        if (data_type == onnx::TensorProto_DataType_INT64) {
            data_vec[i] = static_cast<int64_t>(reinterpret_cast<const int64_t *>(data)[i]);
        } else if (data_type == onnx::TensorProto_DataType_INT32) {
            data_vec[i] = static_cast<int64_t>(reinterpret_cast<const int32_t *>(data)[i]);
        } else {
            DLog("Onnx Converter: do not support tensor proto data type\n");
            assert(0);
        }
    }

    return data_vec;
}

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

    const int input_size = node.input_size();
    if (input_size > 1) {
        onnx::TensorProto tensor = net_info.weights_map[node.input(1)];
        starts                   = get_tensor_data(tensor);
    }

    if (input_size > 2) {
        onnx::TensorProto tensor = net_info.weights_map[node.input(2)];
        ends                     = get_tensor_data(tensor);
    }

    if (input_size > 3) {
        onnx::TensorProto tensor = net_info.weights_map[node.input(3)];
        axes                     = get_tensor_data(tensor);
    }

    if (input_size > 4) {
        onnx::TensorProto tensor = net_info.weights_map[node.input(4)];
        steps                    = get_tensor_data(tensor);
    }

    layer_param << starts.size() << " ";
    for (const auto &start : starts) {
        layer_param << start << " ";
    }
    layer_param << ends.size() << " ";
    for (const auto &end : ends) {
        if (end == LLONG_MAX) {
            layer_param << INT_MAX << " ";
        } else if (end == LLONG_MIN || end == -LLONG_MAX) {
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
