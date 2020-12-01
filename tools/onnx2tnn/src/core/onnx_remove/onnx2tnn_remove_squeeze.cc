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

#include "objseri.h"
#include "onnx2tnn.h"

int Onnx2TNN::RemoveSqueeze(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                            std::map<std::string, onnx::TensorProto>& weights,
                            std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    std::set<std::string> output_node_set;
    int output_node_size = mutable_graph->output_size();
    for (int index = 0; index < output_node_size; index++) {
        const std::string& output_name = mutable_graph->output(index).name();
        if (output_node_set.find(output_name) == output_node_set.end()) {
            output_node_set.emplace(output_name);
        }
    }

    auto const node_count = index_nodes.size();
    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;
        if (node->op_type() != "Squeeze") {
            continue;
        }

        const std::string& node_output_name = node->output(0);
        if (output_node_set.find(node_output_name) != output_node_set.end() && i > 0) {
            bool is_remove = false;
            const auto& node_input_name = node->input(0);
            for (int index = i - 1; index >= 0 && !is_remove; index--) {
                auto pre_node = index_nodes[index].node;
                for (int j = 0; j < pre_node->output_size(); j++) {
                    if (node_input_name == pre_node->output(j)) {
                        pre_node->set_output(j, node_output_name);
                        is_remove = true;
                        break;
                    }
                }
            }
        }

        node->set_op_type(k_tnn_noop_type);
        if (node_reference.find(node->output(0)) == node_reference.end()) {
            continue;
        }
        RemoveIndexNode(index_nodes, i);
        if (node_reference.find(node->output(0)) != node_reference.end()) {
            node_reference.erase(node_reference.find(node->output(0)));
        }
    }
    ClearEmptyNode(index_nodes);
    return 0;
}