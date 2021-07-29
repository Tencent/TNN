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

int Onnx2TNN::RemoveExpand(onnx::GraphProto* mutable_graph,
                              std::vector<IndexNode> & index_nodes,
                              std::map<std::string, onnx::TensorProto>& weights,
                              std::map<std::string, int>& node_reference,
                              std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        //BroadcastNode <= Expand - BroadcastNode
        do {
            std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
            if (next_indexes.size() != 1) {
                break;
            }
            auto node_expand = node;
            auto node_broadcast = index_nodes[next_indexes[0]].node;

            if (node_expand->op_type() != "Expand")
                break;

            if (node_broadcast->op_type() != "Mul" && node_broadcast->op_type() != "Div" &&
                node_broadcast->op_type() != "Add" && node_broadcast->op_type() != "Sub" &&
                node_broadcast->op_type() != "Min" && node_broadcast->op_type() != "Max")
                break;

            // reduce
            node_expand->set_op_type(k_tnn_noop_type);

            node_reference.erase(node_reference.find(node_expand->output(0)));
            blob_names.erase(node_expand->output(0));

            RemoveIndexNode(index_nodes, i);

        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
