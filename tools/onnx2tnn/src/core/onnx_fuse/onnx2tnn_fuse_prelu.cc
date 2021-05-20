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

int Onnx2TNN::FusePRelu(onnx::GraphProto* mutable_graph,
                             std::vector<IndexNode> & index_nodes,
                             std::map<std::string, onnx::TensorProto>& weights,
                             std::map<std::string, int>& node_reference,
                             std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // PReLU <= Unsqueeze - PReLU
        do {
            if (node->op_type() == "Unsqueeze")
            {
                // check weight
                if (weights.find(node->input(0)) == weights.end())
                    continue;

                onnx::TensorProto& B = weights[node->input(0)];
                if (B.dims_size() != 1)
                    continue;

                if (node_reference.find(node->output(0)) == node_reference.end() || node_reference[node->output(0)] != 1)
                    continue;

                // axes = (1, 2)
                std::vector<int64_t> axes = get_node_attr_ai(*node, "axes");
                if (axes.size() != 2)
                    continue;
                if (axes[0] != 1 || axes[1] != 2)
                    continue;

                if (i+1 >= node_count)
                    continue;

                auto node2 = index_nodes[i+1].node;

                if (node2->op_type() != "PRelu")
                    continue;

                if (node2->input(1) != node->output(0))
                    continue;

                // reduce
                node->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node->output(0)));
                blob_names.erase(node->output(0));

                node2->set_input(1, node->input(0));

                i += 1;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
