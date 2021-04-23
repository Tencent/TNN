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

#include <algorithm>

#include "onnx2tnn.h"

int Onnx2TNN::FuseTranspose(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                            std::map<std::string, onnx::TensorProto>& weights,
                            std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;
        // Transpose <= Transpose - Transpose
        do {
            if (node->op_type() == "Transpose" && i + 2 < node_count) {
                auto node1 = index_nodes[i + 1].node;

                if (node1->op_type() != "Transpose")
                    break;
                if (node1->input_size() != 1 || node1->input(0) != node->output(0)) {
                    break;
                }
                
                auto perm = get_node_attr_ai(*node, "perm");
                auto perm1 = get_node_attr_ai(*node1, "perm");
                if (perm.size() != perm1.size()) {
                    break;
                }
                
                auto dst_perm = perm;
                for (int i=0; i < perm.size(); i++) {
                    dst_perm[i] = perm[perm1[i]];
                }
                
                node->set_output(0, node1->output(0));
                set_node_attr_ai(*node, "perm", dst_perm);
                perm = get_node_attr_ai(*node, "perm");
                // reduce
                node1->set_op_type(k_tnn_noop_type);
                
                
                node_reference.erase(node_reference.find(node1->output(0)));
                blob_names.erase(node->output(0));

                i += 2;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
