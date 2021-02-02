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

int Onnx2TNN::RemovePad(onnx::GraphProto* mutable_graph,
                              std::vector<IndexNode> & index_nodes,
                              std::map<std::string, onnx::TensorProto>& weights,
                              std::map<std::string, int>& node_reference,
                              std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        //x <= x - Pad(1x1s1)
        do {
            if (i + 1 >= node_count) {
                break;
            }
            auto node_pad = index_nodes[i+1].node;
            if (node_pad->op_type() != "Pad")
                break;
            
            if (node_reference.find(node_pad->output(0)) == node_reference.end() ||
                node_reference[node_pad->output(0)] != 1)
                break;

            bool need_remove = true;
            std::vector<int64_t> pads =
                get_node_attr_ai(*node_pad, "pads", weights, 1);
            for (auto item : pads) {
                need_remove = need_remove && (item == 0);
            }

            if (!need_remove) {
                break;
            }
            // reduce
            node_pad->set_op_type(k_tnn_noop_type);

            node_reference.erase(node_reference.find(node_pad->output(0)));
            blob_names.erase(node_pad->output(0));

            RemoveIndexNode(index_nodes, i+1);
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
