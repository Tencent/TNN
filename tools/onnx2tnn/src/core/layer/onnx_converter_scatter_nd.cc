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

DECLARE_OP_CONVERTER(ScatterND);

string OnnxOpConverterScatterND::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "ScatterND";
}

string OnnxOpConverterScatterND::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    ostringstream layer_param;
    return layer_param.str();
}

int OnnxOpConverterScatterND::WriteTNNModel(serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    const std::string& onnx_op = node.op_type();
    std::string name = !node.name().empty() ? node.name() : node.output(0);
    const std::string& tnn_layer_type = TNNOpType(node, net_info);

    net_writer->put_int(0);
    net_writer->put_string(tnn_layer_type);
    net_writer->put_string(name);

    net_writer->put_string(name);
    const onnx::TensorProto& indices = net_info.weights_map[node.input(1)];

    // save indices shape
    // cast int64_t to int
    std::vector<int> indices_dims(indices.dims().begin(), indices.dims().end());
    net_writer->put_dims(indices_dims);
    // save indices value
    WriteTensorData(indices, net_writer, DATA_TYPE_INT32);

    // save update
    int has_update = 0;
    auto& update_name = node.input(2);
    if (net_info.weights_map.find(update_name) != net_info.weights_map.end()) {
        has_update = 1;
        net_writer->put_int(has_update);
        const onnx::TensorProto& update = net_info.weights_map[update_name];
        std::vector<int> update_dims(update.dims().begin(), update.dims().end());
        net_writer->put_dims(update_dims);
        WriteTensorData(update, net_writer, DATA_TYPE_FLOAT);
    } else {
        net_writer->put_int(has_update);
    }

    return 0;
}

REGISTER_OP_CONVERTER(ScatterND, ScatterND);
