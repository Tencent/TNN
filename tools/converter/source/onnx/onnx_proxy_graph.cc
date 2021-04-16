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

#include "onnx_proxy_graph.h"

#include "onnx_utils.h"
namespace TNN_CONVERTER {

OnnxProxyNode::OnnxProxyNode() : op_name(), op_type(), onnx_node(nullptr) {
    // do nothing
}

OnnxProxyNode::~OnnxProxyNode() {
    // do nothing
}

OnnxProxyGraph::OnnxProxyGraph(const onnx::GraphProto* graph_proto) {
    this->graph_proto_ = graph_proto;
    InitProxyGraph();
}

OnnxProxyGraph::~OnnxProxyGraph() {
    // do nothing
}

void OnnxProxyGraph::InitProxyGraph() {
    const int node_size = this->graph_proto_->node_size();
    for (int i = 0; i < node_size; ++i) {
        const auto& onnx_node = this->graph_proto_->node(i);
        if (onnx_node.op_type() == "Constant") {
            // TODO
            auto tensor = GetTensorFromConstantNode(onnx_node);
            proxy_initializers_map_.insert(std::make_pair(onnx_node.output(0), tensor));
        } else {
            std::shared_ptr<OnnxProxyNode> proxy_node(new OnnxProxyNode());
            proxy_node->op_name   = onnx_node.output(0);
            proxy_node->op_type   = onnx_node.op_type();
            proxy_node->onnx_node = &onnx_node;
            proxy_nodes_map_.insert(std::make_pair(onnx_node.output(0), proxy_node));
        }
    }

    const int initializer_size = this->graph_proto_->initializer_size();
    for (int i = 0; i < initializer_size; ++i) {
        const auto& initializer = this->graph_proto_->initializer(i);
        proxy_initializers_map_.insert(std::make_pair(initializer.name(), &initializer));
    }

    const int input_size = this->graph_proto_->input_size();
    for (int i = 0; i < input_size; ++i) {
        const auto& input = this->graph_proto_->input(i);
        proxy_inputs_map_.insert(std::make_pair(input.name(), &input));
    }

    const int output_size = this->graph_proto_->output_size();
    for (int i = 0; i < output_size; ++i) {
        const auto& output = this->graph_proto_->output(i);
        proxy_outputs_map_.insert(std::make_pair(output.name(), &output));
    }
}
}  // namespace TNN_CONVERTER
