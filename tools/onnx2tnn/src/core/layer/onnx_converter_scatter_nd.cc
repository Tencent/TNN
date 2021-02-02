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

string OnnxOpConverterScatterND::TNNOpType(NodeProto& node, OnnxNetInfo& net_info) {
    return "ScatterND";
}

string OnnxOpConverterScatterND::TNNLayerParam(NodeProto& node, OnnxNetInfo& net_info) {
    ostringstream layer_param;
    return layer_param.str();
}

bool OnnxOpConverterScatterND::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return true;
};

int OnnxOpConverterScatterND::WriteTNNModel(Serializer* net_writer, NodeProto& node, OnnxNetInfo& net_info) {
    const std::string& onnx_op        = node.op_type();
    std::string name                  = !node.name().empty() ? node.name() : node.output(0);
    const std::string& tnn_layer_type = TNNOpType(node, net_info);

    net_writer->PutInt(0);
    net_writer->PutString(tnn_layer_type);
    net_writer->PutString(name);

    const auto& weights_map  = net_info.weights_map;
    const auto& indices_name = node.input(1);
    bool has_indices         = false;
    if (weights_map.find(node.input(1)) != weights_map.end()) {
        has_indices                      = true;
        const onnx::TensorProto& indices = net_info.weights_map[node.input(1)];
        // save indices shape
        // cast int64_t to int
        std::vector<int> indices_dims(indices.dims().begin(), indices.dims().end());
        net_writer->PutBool(has_indices);
        // save indices value
        WriteIntTensorData(indices, net_writer);
    } else {
        net_writer->PutBool(has_indices);
    }
    // save update
    bool has_update   = false;
    auto& update_name = node.input(2);
    if (weights_map.find(update_name) != weights_map.end()) {
        has_update                      = true;
        const onnx::TensorProto& update = net_info.weights_map[update_name];
        std::vector<int> update_dims(update.dims().begin(), update.dims().end());
        net_writer->PutBool(has_update);
        WriteTensorData(update, net_writer, DATA_TYPE_FLOAT);
    } else {
        net_writer->PutBool(has_update);
    }

    return 1;
}

REGISTER_OP_CONVERTER(ScatterND, ScatterND);
