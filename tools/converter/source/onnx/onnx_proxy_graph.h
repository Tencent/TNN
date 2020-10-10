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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_ONNX_ONNX_PROXY_GRAPH_H_
#define TNN_TOOLS_CONVERTER_SOURCE_ONNX_ONNX_PROXY_GRAPH_H_
#include <memory>
#include <set>

#include "onnx.pb.h"

namespace TNN_CONVERTER {

class OnnxProxyNode {
public:
    OnnxProxyNode();
    ~OnnxProxyNode();
    std::string op_name;
    std::string op_type;
    const onnx::NodeProto* onnx_node;

    std::vector<std::string> in_edges;
    std::vector<std::string> out_edges;
};

class OnnxProxyGraph {
public:
    OnnxProxyGraph(const onnx::GraphProto* graph_proto);
    OnnxProxyGraph() = delete;
    ~OnnxProxyGraph();

    const onnx::GraphProto* graph_proto_;
    std::map<std::string, std::shared_ptr<OnnxProxyNode>> proxy_nodes_map_;
    std::map<std::string, const onnx::TensorProto*> proxy_initializers_map_;
    std::map<std::string, const onnx::ValueInfoProto*> proxy_inputs_map_;
    std::map<std::string, const onnx::ValueInfoProto*> proxy_outputs_map_;
    std::set<std::string> constant_node_to_delete_;

private:
    void InitProxyGraph();
};

}  // namespace TNN_CONVERTER
#endif  // TNN_TOOLS_CONVERTER_SOURCE_ONNX_ONNX_PROXY_GRAPH_H_
