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

int Onnx2TNN::FuseBatchNorm(onnx::GraphProto* mutable_graph,
                                 std::vector<IndexNode> & index_nodes,
                                 std::map<std::string, onnx::TensorProto>& weights,
                                 std::map<std::string, int>& node_reference,
                                 std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // BatchNormalization <= Unsqueeze - BatchNormalization - Squeeze
        do {
            if (node->op_type() == "Unsqueeze")
            {
                if (node_reference.find(node->output(0)) == node_reference.end() || node_reference[node->output(0)] != 1)
                    continue;

                if (i+2 >= node_count)
                    continue;

                auto node2 = index_nodes[i+1].node;
                auto node3 = index_nodes[i+2].node;

                if (node2->op_type() != "BatchNormalization" || node3->op_type() != "Squeeze")
                    continue;

                if (node_reference.find(node2->output(0)) == node_reference.end() || node_reference[node2->output(0)] != 1)
                    continue;

                if (node2->input(0) != node->output(0) || node3->input(0) != node2->output(0))
                    continue;

                // reduce
                node->set_op_type(k_tnn_noop_type);
                node3->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node2->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node2->output(0));

                node2->set_input(0, node->input(0));
                node2->set_output(0, node3->output(0));

                i += 2;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
