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

DECLARE_OP_CONVERTER_WITH_FUNC(Scatter, virtual std::vector<std::string> GetValidInputNames(NodeProto& node,
                                                                                            OnnxNetInfo& net_info););

string OnnxOpConverterScatter::TNNOpType(NodeProto& node, OnnxNetInfo& net_info) {
    return "Scatter";
}

string OnnxOpConverterScatter::TNNLayerParam(NodeProto& node, OnnxNetInfo& net_info) {
    int axis = 0;
    if (node_has_attr(node, "axis")) {
        axis = (int)get_node_attr_i(node, "axis", 0);
    }

    ostringstream layer_param;
    layer_param << axis << " ";

    return layer_param.str();
}

std::vector<std::string> OnnxOpConverterScatter::GetValidInputNames(NodeProto& node, OnnxNetInfo& net_info) {
    auto iter_indices = net_info.weights_map.find(node.input(1));
    auto iter_updates = net_info.weights_map.find(node.input(2));
    if (iter_indices == net_info.weights_map.end() && iter_updates == net_info.weights_map.end()) {
        // X, Y, condition order for input
        return {node.input(0), node.input(1), node.input(2)};
    } else if (iter_updates == net_info.weights_map.end()) {
        return {node.input(0), node.input(2)};
    } else {
        return {node.input(0), node.input(1)};
    }
}

bool OnnxOpConverterScatter::HasLayerResource(NodeProto& node, OnnxNetInfo& net_info) {
    // support indices input is not const, move layer resource to const blob
    auto iter_indices = net_info.weights_map.find(node.input(1));
    auto iter_updates = net_info.weights_map.find(node.input(2));
    if (iter_indices == net_info.weights_map.end() && iter_updates == net_info.weights_map.end()) {
        return false;
    } else {
        return true;
    }
};

int OnnxOpConverterScatter::WriteTNNModel(Serializer* net_writer, NodeProto& node, OnnxNetInfo& net_info) {
    // support indices input is not const, move layer resource to const blob
    if (!HasLayerResource(node, net_info)) {
        return 0;
    }

    const std::string& onnx_op        = node.op_type();
    std::string name                  = !node.name().empty() ? node.name() : node.output(0);
    const std::string& tnn_layer_type = TNNOpType(node, net_info);

    net_writer->PutInt(0);
    net_writer->PutString(tnn_layer_type);
    net_writer->PutString(name);

    const auto& weights_map  = net_info.weights_map;
    const auto& indices_name = node.input(1);
    bool has_indices         = false;
    if (weights_map.find(indices_name) != weights_map.end()) {
        has_indices                      = true;
        const onnx::TensorProto& indices = net_info.weights_map[indices_name];
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

REGISTER_OP_CONVERTER(Scatter, Scatter);
