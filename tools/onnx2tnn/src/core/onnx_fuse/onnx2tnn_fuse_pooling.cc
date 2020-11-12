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

#include "half_utils.h"
#include "objseri.h"
#include "onnx2tnn.h"
#include "onnx_utility.h"

int Onnx2TNN::FusePooling(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                          std::map<std::string, onnx::TensorProto>& weights, std::map<std::string, int>& node_reference,
                          std::set<std::string>& blob_names) {
    //  Pooling <= Pad + Pooling
    auto const node_count = index_nodes.size();
    for (int i = 0; i < node_count; i++) {
        auto node_pad = index_nodes[i].node;
        do {
            if (node_pad->op_type() == "Pad" && i + 1 < node_count) {
                auto node_pooling = index_nodes[i + 1].node;
                if (node_pooling->op_type() != "MaxPool" && node_pooling->op_type() != "GlobalMaxPool") {
                    break;
                }
                if (node_pad->output(0) != node_pooling->input(0)) {
                    break;
                }
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }
                auto pads = get_node_attr_ai(*node_pad, "pads", weights, 1);
                if (pads.size() != 8) {
                    break;
                }
                auto pooling_pads = get_node_attr_ai(*node_pooling, "pads");
                if (pooling_pads.size() != 4) {
                    break;
                }
                if ((pads[2] != pads[6]) || (pads[3] != pads[7])) {
                    break;
                }
                pooling_pads[0] += pads[2];
                pooling_pads[1] += pads[3];
                pooling_pads[2] += pads[6];
                pooling_pads[3] += pads[7];
                set_node_attr_ai(*node_pooling, "pads", pooling_pads);
                node_pad->set_op_type(k_tnn_noop_type);
                node_pooling->set_input(0, node_pad->input(0));
                node_reference.erase(node_reference.find(node_pad->output(0)));
                i += 1;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
