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
                
                auto axes = get_node_attr_ai(*node_suqeeze, "axes", weights, 1);
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

        // LSTM <= Reshape - Unsqueeze - Reshape - Unsqueeze - LSTM(direction=forward)
        // - Squeeze(axis = 1) - Squeeze - Reshape - Squeeze - Reshape
        do {
            auto tp = node->op_type();
            auto on = node->output(0);
            if (node->op_type() == "LSTM" && i + 1 < node_count) {
                onnx::NodeProto* node_lstm = node;
                auto direction             = get_node_attr_s(*node_lstm, "direction", "forward");
                if (direction != "forward" && direction != "reverse") {
                    break;
                }

                const auto& R_name = node->input(2);
                if (weights.find(R_name) == weights.end()) {
                    break;
                }
                const auto R          = weights[R_name];
                const int hidden_size = R.dims(2);

                std::vector<int> previous_indexs = GetPreviousIndexNode(index_nodes, i);
                std::vector<int> next_indexes    = GetNextIndexNode(index_nodes, i);

                if (next_indexes.size() != 3) {
                    break;
                }

                // check unsqueeze is valid
                auto check_unsqueeze = [&](onnx::NodeProto* node,
                                           std::map<std::string, onnx::TensorProto>& weights) -> bool {
                    if (node->op_type() != "Unsqueeze") {
                        return false;
                    }
                    auto axes = get_node_attr_ai(*node, "axes", weights, 1);
                    if (axes.size() != 1 || axes[0] != 0) {
                        return false;
                    }

                    return true;
                };

                // Let the index of unsqueeze at the end
                {
                    const auto tmp_indexes = previous_indexs;
                    int unsqueeze_index    = previous_indexs.size() - 1;
                    int other_index        = 0;
                    for (const auto item : tmp_indexes) {
                        const auto& op_type = index_nodes[item].node->op_type();
                        if (op_type != "Unsqueeze") {
                            previous_indexs[other_index++] = item;
                        } else {
                            previous_indexs[unsqueeze_index--] = item;
                        }
                    }
                }

                onnx::NodeProto* node_unsqueeze0 = index_nodes[previous_indexs[1]].node;
                onnx::NodeProto* node_unsqueeze1 = index_nodes[previous_indexs[2]].node;
                if (!check_unsqueeze(node_unsqueeze0, weights) || !check_unsqueeze(node_unsqueeze1, weights)) {
                    break;
                }

                std::vector<int> unsqueeze0_previous_indexs = GetPreviousIndexNode(index_nodes, previous_indexs[1]);
                std::vector<int> unsqueeze1_previous_indexs = GetPreviousIndexNode(index_nodes, previous_indexs[2]);
                if (unsqueeze0_previous_indexs.size() > 2 || unsqueeze1_previous_indexs.size() > 2) {
                    break;
                }

                // check reshape is valid
                auto check_reshape = [&](onnx::NodeProto* node, std::vector<IndexNode>& index_nodes,
                                         std::map<std::string, onnx::TensorProto>& weights, int index,
                                         int target_shape_dims, int hidden_size) -> bool {
                    if (node->op_type() != "Reshape") {
                        return false;
                    }
                    std::vector<int> reshape_next_indexes = GetNextIndexNode(index_nodes, index);
                    if (reshape_next_indexes.size() != 1) {
                        return false;
                    }
                    auto shape = get_node_attr_ai(*node, "shape", weights, 1);
                    if (shape.size() != target_shape_dims) {
                        return false;
                    }

                    if (shape.back() != hidden_size) {
                        return false;
                    }

                    return true;
                };

                onnx::NodeProto* node_reshape0 = index_nodes[unsqueeze0_previous_indexs[0]].node;
                onnx::NodeProto* node_reshape1 = index_nodes[unsqueeze1_previous_indexs[0]].node;
                if (!check_reshape(node_reshape0, index_nodes, weights, unsqueeze0_previous_indexs[0], 2,
                                   hidden_size) ||
                    !check_reshape(node_reshape1, index_nodes, weights, unsqueeze1_previous_indexs[0], 2,
                                   hidden_size)) {
                    break;
                }

                onnx::NodeProto* node_squeeze0 = index_nodes[next_indexes[0]].node;

                // check op
                if (!(node_squeeze0->op_type() == "Squeeze"))
                    break;

                auto axes = get_node_attr_ai(*node_squeeze0, "axes", weights, 1);
                if (axes.size() != 1 || axes[0] != 1)
                    break;

                auto check_squeeze = [&](onnx::NodeProto* node, std::map<std::string, onnx::TensorProto>& weights) {
                    if (node->op_type() != "Squeeze") {
                        return false;
                    }
                    auto axes = get_node_attr_ai(*node, "axes", weights, 1);
                    if (!axes.empty()) {
                        return false;
                    }

                    return true;
                };

                onnx::NodeProto* node_squeeze1 = index_nodes[next_indexes[1]].node;
                onnx::NodeProto* node_squeeze2 = index_nodes[next_indexes[2]].node;
                if (!check_squeeze(node_squeeze1, weights) || !check_squeeze(node_squeeze2, weights)) {
                    break;
                }

                std::vector<int> squeeze1_next_index = GetNextIndexNode(index_nodes, next_indexes[1]);
                std::vector<int> squeeze2_next_index = GetNextIndexNode(index_nodes, next_indexes[2]);
                if (squeeze1_next_index.size() != 1 && squeeze2_next_index.size() != 2) {
                    break;
                }

                onnx::NodeProto* node_reshape2 = index_nodes[squeeze1_next_index[0]].node;
                onnx::NodeProto* node_reshape3 = index_nodes[squeeze2_next_index[0]].node;
                if (!check_reshape(node_reshape2, index_nodes, weights, squeeze1_next_index[0], 3, hidden_size) ||
                    !check_reshape(node_reshape3, index_nodes, weights, squeeze2_next_index[0], 3, hidden_size)) {
                    break;
                }

                node_reshape0->set_op_type(k_tnn_noop_type);
                node_reshape1->set_op_type(k_tnn_noop_type);
                node_squeeze0->set_op_type(k_tnn_noop_type);
                node_squeeze1->set_op_type(k_tnn_noop_type);
                node_squeeze2->set_op_type(k_tnn_noop_type);
                node_reshape2->set_op_type(k_tnn_noop_type);
                node_reshape3->set_op_type(k_tnn_noop_type);

                node_unsqueeze0->set_op_type("Squeeze");
                const auto node_unsqueeze0_input = node_unsqueeze0->input(0);
                node_unsqueeze0->clear_input();
                node_unsqueeze0->add_input(node_unsqueeze0_input);
                onnx::AttributeProto* unsqueeze0_attr = node_unsqueeze0->add_attribute();
                unsqueeze0_attr->set_name("axes");
                unsqueeze0_attr->add_ints(-1);

                node_unsqueeze1->set_op_type("Squeeze");
                const auto node_unsqueeze1_input = node_unsqueeze1->input(0);
                node_unsqueeze1->clear_input();
                node_unsqueeze1->add_input(node_unsqueeze1_input);
                onnx::AttributeProto* unsqueeze1_attr = node_unsqueeze1->add_attribute();
                unsqueeze1_attr->set_name("axes");
                unsqueeze1_attr->add_ints(-1);

                node_reference.erase(node_reference.find(node_lstm->output(0)));
                node_reference.erase(node_reference.find(node_lstm->output(1)));
                node_reference.erase(node_reference.find(node_lstm->output(2)));
                node_reference.erase(node_reference.find(node_reshape0->output(0)));
                node_reference.erase(node_reference.find(node_reshape1->output(0)));
                node_reference.erase(node_reference.find(node_unsqueeze0->output(0)));
                node_reference.erase(node_reference.find(node_unsqueeze1->output(0)));
                node_reference.erase(node_reference.find(node_squeeze1->output(0)));
                node_reference.erase(node_reference.find(node_squeeze2->output(0)));

                blob_names.erase(node_lstm->output(0));
                blob_names.erase(node_lstm->output(1));
                blob_names.erase(node_lstm->output(2));
                blob_names.erase(node_reshape0->output(0));
                blob_names.erase(node_reshape1->output(0));
                blob_names.erase(node_unsqueeze0->output(0));
                blob_names.erase(node_unsqueeze1->output(0));
                blob_names.erase(node_squeeze0->output(0));
                blob_names.erase(node_squeeze1->output(0));

                node_unsqueeze0->set_input(0, node_reshape0->input(0));
                node_unsqueeze1->set_input(0, node_reshape1->input(0));
                node_lstm->set_output(0, node_squeeze0->output(0));
                node_lstm->set_output(1, node_reshape2->output(0));
                node_lstm->set_output(2, node_reshape3->output(0));

                node_squeeze0->clear_input();
                node_squeeze0->clear_output();
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
