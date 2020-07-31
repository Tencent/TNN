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

#include "objseri.h"
#include "onnx2tnn.h"

int Onnx2TNN::FuseRoiAlign(onnx::GraphProto* mutable_graph,
                       std::vector<IndexNode>& index_nodes,
                       std::map<std::string, onnx::TensorProto>& weights,
                       std::map<std::string, int>& node_reference,
                       std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    // RoiAlign <= Gather - Squeeze - Cast -Gather -RoiAlign
    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        do {
            std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
            if (next_indexes.size() != 2 || i + 5 >= node_count) {
                break;
            }

            auto roi_align = index_nodes[i + 5].node;
            if (roi_align->op_type() != "RoiAlign") {
                break;
            }

            for (int idx = 1; idx < 5; idx++) {
                auto tmp_node = index_nodes[i + idx].node;
                tmp_node->set_op_type(k_tnn_noop_type);
                node_reference.erase(node_reference.find(tmp_node->output(0)));
                blob_names.erase(node->output(0));
            }

            roi_align->set_input(1, node->output(0));
            i += 5;

        } while (0);
    }
}
