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

#include <math.h>


#include "onnx2tnn.h"

int Onnx2TNN::RemoveConcat(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                           std::map<std::string, onnx::TensorProto>& weights,
                           std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        // x <= x - Concat(1 input)
        do {
            auto node = index_nodes[i].node;
            if (i + 1 >= node_count) {
                break;
            }
            auto node_concat = index_nodes[i + 1].node;
            if (node_concat->op_type() != "Concat")
                break;

            if (node_reference.find(node->output(0)) == node_reference.end() || node_reference[node->output(0)] != 1)
                break;

            if (node_concat->input_size() > 1) {
                break;
            }

            // reduce
            node_concat->set_op_type(k_tnn_noop_type);

            auto erase_node = node_reference.find(node_concat->output(0));
            if (erase_node != node_reference.end()) {
                node_reference.erase(node_reference.find(node_concat->output(0)));
            }
            blob_names.erase(node->output(0));

            //            node->set_output(0, node_concat->output(0));
            RemoveIndexNode(index_nodes, i + 1);
        } while (0);

        // X -> Concat(1 input) -> Y = X -> Y
        do {
            if (i + 1 >= node_count) {
                break;
            }
            auto node_concat = index_nodes[i].node;
            auto node_next   = index_nodes[i + 1].node;
            if (node_concat->op_type() != "Concat")
                break;

            if (node_reference.find(node_concat->output(0)) == node_reference.end() ||
                node_reference[node_concat->output(0)] != 1)
                break;

            if (node_concat->input_size() > 1) {
                break;
            }

            // reduce
            node_concat->set_op_type(k_tnn_noop_type);
            for (int j = 0; j < node_next->input_size(); j++) {
                std::string node_next_input_name = node_next->input(j);
                if (node_concat->output(0) == node_next_input_name) {
                    node_next->set_input(j, node_concat->input(0));
                }
            }
            node_reference.erase(node_reference.find(node_concat->output(0)));
            blob_names.erase(node_concat->output(0));

            //            node->set_output(0, node_concat->output(0));
            RemoveIndexNode(index_nodes, i);
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
