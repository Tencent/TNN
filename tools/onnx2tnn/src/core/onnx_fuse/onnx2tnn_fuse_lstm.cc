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

int Onnx2TNN::FuseLSTM(onnx::GraphProto* mutable_graph,
                                  std::vector<IndexNode> & index_nodes,
                                  std::map<std::string, onnx::TensorProto>& weights,
                                  std::map<std::string, int>& node_reference,
                                  std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // LSTM <= LSTM(direction=forward) - Squeeze(axis = 1)
        do {
            if (node->op_type() == "LSTM" && i + 1 < node_count) {
                onnx::NodeProto* node_lstm = node;
                auto direction = get_node_attr_s(*node_lstm, "direction", "forward");
                if (direction != "forward" && direction != "reverse") {
                    break;
                }
                
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }
                onnx::NodeProto* node_suqeeze = index_nodes[next_indexes[0]].node;
                
                // check op
                if (!(node_suqeeze->op_type() == "Squeeze"))
                    break;
                
                auto axes = get_node_attr_ai(*node_suqeeze, "axes");
                if (axes.size() != 1 || axes[0] != 1)
                        break;
                
                node_suqeeze->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node_lstm->output(0)));
                blob_names.erase(node_lstm->output(0));
                
                node_lstm->set_output(0, node_suqeeze->output(0));

                i += 1;
            }
        } while (0);
        // LSTM <= LSTM(direction=bidirectional) - Transpose - Reshape
        do {
            if (node->op_type() == "LSTM" && i + 2 < node_count) {
                onnx::NodeProto* node_lstm = node;
                auto direction = get_node_attr_s(*node_lstm, "direction", "forward");
                if (direction != "bidirectional") {
                    break;
                }
                
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }
                onnx::NodeProto* node_transpose = index_nodes[next_indexes[0]].node;
                
                // check op
                if (node_transpose->op_type() != "Transpose")
                    break;
                auto perm = get_node_attr_ai(*node_transpose, "perm");
                if (perm.size() != 4 || perm[0] != 0 || perm[1] != 2 || perm[2] != 1 || perm[3] != 3)
                        break;
                
                next_indexes = GetNextIndexNode(index_nodes, next_indexes[0]);
                if (next_indexes.size() != 1) {
                    break;
                }
                onnx::NodeProto* node_reshape = index_nodes[next_indexes[0]].node;
                // check op
                if (node_reshape->op_type() != "Reshape")
                    break;
                auto shape = get_node_attr_ai(*node_reshape, "shape", onnx_net_info_, 1);
                if (shape.size() != 3 || shape[0] != 0 || shape[1] != 0 || shape[2] != -1)
                        break;
                
                node_transpose->set_op_type(k_tnn_noop_type);
                node_reshape->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node_lstm->output(0)));
                blob_names.erase(node_lstm->output(0));
                node_reference.erase(node_reference.find(node_transpose->output(0)));
                blob_names.erase(node_transpose->output(0));
                
                node_lstm->set_output(0, node_reshape->output(0));

                i += 2;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
