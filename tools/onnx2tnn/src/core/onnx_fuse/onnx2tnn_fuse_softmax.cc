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

int Onnx2TNN::FuseSoftmax(onnx::GraphProto* mutable_graph,
                               std::vector<IndexNode> & index_nodes,
                               std::map<std::string, onnx::TensorProto>& weights,
                               std::map<std::string, int>& node_reference,
                               std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // Softmax <= Exp - ReduceSum - Div
        do {
            if (node->op_type() == "Exp" && i + 2 < node_count) {
                auto node_exp = node;
                auto node_reducesum = index_nodes[i+1].node;
                auto node_div = index_nodes[i+2].node;

                // check op
                if (!(node_reducesum->op_type() == "ReduceSum" &&
                    node_div->op_type() == "Div"))
                    break;
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 2) {
                    break;
                }
                next_indexes = GetNextIndexNode(index_nodes, i+1);
                if (next_indexes.size() != 1) {
                    break;
                }

                auto axis = get_node_attr_ai(*node_reducesum, "axes");

                bool can_fuse = false;
                int softmax_axis = 1;
                if (axis.size() == 1 &&
                    node_div->input_size() == 2) {
                    softmax_axis = axis[0];

                    can_fuse =
                        (node_div->input(0) == node_exp->output(0) &&
                        node_div->input(1) == node_reducesum->output(0)) ||
                        (node_div->input(1) == node_exp->output(0) &&
                        node_div->input(0) == node_reducesum->output(0));
                } else {
                    DLog("axis size %d, axis[0]:%d input_size:%d\n", (int)axis.size(), (int)axis[0], (int)node_div->input_size());
                }

                if (!can_fuse) {
                    DLog("exp didn't fuse to softmax\n");
                    break;
                }

                node_reducesum->set_op_type(k_tnn_noop_type);
                node_div->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node_exp->output(0)));
                node_reference.erase(
                    node_reference.find(node_reducesum->output(0)));
                blob_names.erase(node_exp->output(0));
                blob_names.erase(node_reducesum->output(0));

                node_exp->set_op_type("Softmax");
                node_exp->set_output(0, node_div->output(0));
                onnx::AttributeProto* attr_group = node_exp->add_attribute();
                attr_group->set_name("axis");
                attr_group->set_i(softmax_axis);

                i += 2;

            }
        } while (0);

        // Softmax <= Transpose - Softmax - Transpose
        do {
          if (node->op_type() == "Transpose" && i + 2 < node_count) {
              auto node_transpose1 = node;
              auto node_softmax = index_nodes[i+1].node;
              auto node_transpose2 = index_nodes[i+2].node;

              // check op
              if (!(node_softmax->op_type() == "Softmax" &&
                    node_transpose2->op_type() == "Transpose"))
                  break;

              std::vector<int64_t> perm1 =
                  get_node_attr_ai(*node_transpose1, "perm");
              int64_t axis = get_node_attr_i(*node_softmax, "axis", 1);
              std::vector<int64_t> perm2 =
                  get_node_attr_ai(*node_transpose2, "perm");
              bool can_fuse = false;
              if (perm1.size() == 4 && perm2.size() == 4) {
                  can_fuse = axis == 3 && perm1[0] == 0 && perm1[1] == 2 &&
                             perm1[2] == 3 && perm1[3] == 1 &&
                             perm2[0] == 0 && perm2[1] == 3 &&
                             perm2[2] == 1 && perm2[3] == 2;
              }

              if (!can_fuse) {
                  break;
              }

              node_transpose1->set_op_type(k_tnn_noop_type);
              node_transpose2->set_op_type(k_tnn_noop_type);

              node_reference.erase(
                  node_reference.find(node_transpose1->output(0)));
              node_reference.erase(
                  node_reference.find(node_softmax->output(0)));
              blob_names.erase(node_transpose1->output(0));
              blob_names.erase(node_softmax->output(0));

              auto axis_attr = get_node_mutable_attr(*node_softmax, "axis");
              axis_attr->set_i(1);
              node_softmax->set_input(0, node_transpose1->input(0));
              node_softmax->set_output(0, node_transpose2->output(0));

              i += 2;
          }
        } while (0);

        // Softmax <= Transpose - Reshape - Softmax - Reshape - Transpose
        do {
            if (node->op_type() == "Transpose" && i + 4 < node_count) {
              auto node_transpose1 = node;
              auto node_reshape1 = index_nodes[i+1].node;
              auto node_softmax = index_nodes[i+2].node;
              auto node_reshape2 = index_nodes[i+3].node;
              auto node_transpose2 = index_nodes[i+4].node;

              // check op
              if (!(node_reshape1->op_type() == "Reshape" &&
                    node_softmax->op_type() == "Softmax" &&
                    node_reshape2->op_type() == "Reshape" &&
                    node_transpose2->op_type() == "Transpose"))
                  break;

              std::vector<int64_t> perm1 =
                  get_node_attr_ai(*node_transpose1, "perm");
              int64_t axis = get_node_attr_i(*node_softmax, "axis", 1);
              std::vector<int64_t> perm2 =
                  get_node_attr_ai(*node_transpose2, "perm");
              bool can_fuse = false;
              if (perm1.size() == 4 && perm2.size() == 4) {
                  can_fuse = axis == 1 && perm1[0] == 0 && perm1[1] == 2 &&
                             perm1[2] == 3 && perm1[3] == 1 &&
                             perm2[0] == 0 && perm2[1] == 3 &&
                             perm2[2] == 1 && perm2[3] == 2;
              }

              if (!can_fuse) {
                  break;
              }

              node_transpose1->set_op_type(k_tnn_noop_type);
              node_reshape1->set_op_type(k_tnn_noop_type);
              node_reshape2->set_op_type(k_tnn_noop_type);
              node_transpose2->set_op_type(k_tnn_noop_type);

              node_reference.erase(node_transpose1->output(0));
              node_reference.erase(node_reshape1->output(0));
              node_reference.erase(node_softmax->output(0));
              node_reference.erase(node_reshape2->output(0));

              blob_names.erase(node_transpose1->output(0));
              blob_names.erase(node_reshape1->output(0));
              blob_names.erase(node_softmax->output(0));
              blob_names.erase(node_reshape2->output(0));

              auto axis_attr = get_node_mutable_attr(*node_softmax, "axis");
              axis_attr->set_i(1);
              node_softmax->set_input(0, node_transpose1->input(0));
              node_softmax->set_output(0, node_transpose2->output(0));

              i += 4;
          }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
