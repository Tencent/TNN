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

DECLARE_OP_CONVERTER(Const);

string OnnxOpConverterConst::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "Const";
}

string OnnxOpConverterConst::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    onnx::TensorProto tensor = get_node_attr_tensor(node, "value");
    std::vector<int> dims = CreateDimsVectorFromTensor(tensor);
    ostringstream layer_param;
    layer_param << dims.size() << " ";
    for (const auto& dim: dims) {
        layer_param << dim << " ";
    }
    return layer_param.str();
}

bool OnnxOpConverterConst::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return true;
}

int OnnxOpConverterConst::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    const std::string &onnx_op        = node.op_type();
    std::string name                  = !node.name().empty() ? node.name() : node.output(0);
    const std::string &tnn_layer_type = TNNOpType(node, net_info);

    net_writer->PutInt(0);  //触发type from string
    net_writer->PutString(tnn_layer_type);
    net_writer->PutString(name);

    onnx::TensorProto tensor = get_node_attr_tensor(node, "value");
    WriteTensorData(tensor, net_writer, net_info.data_type);
    return 1;
}

REGISTER_OP_CONVERTER(Const, Constant);
