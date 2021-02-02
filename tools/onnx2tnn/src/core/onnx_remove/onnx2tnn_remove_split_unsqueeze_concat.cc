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

int Onnx2TNN::RemoveSplitUnsqueezeConcat(onnx::GraphProto* mutable_graph,
                                              std::vector<IndexNode> & index_nodes,
                                              std::map<std::string, onnx::TensorProto>& weights,
                                              std::map<std::string, int>& node_reference,
                                              std::set<std::string>& blob_names) {
    const int node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        onnx::NodeProto* node = index_nodes[i].node;

        //x <= x - Split - Unsquzze(n) - Concat
        do {
            if (i + 3 >= node_count) {
                break;
            }
            auto node_split = index_nodes[i + 1].node;
            if (node_split->op_type() != "Split")
                break;

            auto node_unqueeze = index_nodes[i + 2].node;
            int index_concat = i + 3;
            auto node_concat = index_nodes[index_concat].node;
            while (node_concat->op_type() == "Unsqueeze" && index_concat+1 < node_count) {
                node_concat = index_nodes[++index_concat].node;
            }

            if (node_split->op_type() != "Split" ||
                node_unqueeze->op_type() != "Unsqueeze" ||
                node_concat->op_type() != "Concat")
                break;

            // reduce
            for (int index=i+1; index<=index_concat; index++) {
                index_nodes[index].node->set_op_type(k_tnn_noop_type);
                auto item = node_reference.find(index_nodes[index].node->output(0));
                if (item != node_reference.end()) {
                    node_reference.erase(item);
                }

                blob_names.erase(index_nodes[index].node->output(0));
            }

            node->set_output(0, node_concat->output(0));
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
