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

int Onnx2TNN::FuseFlatten(onnx::GraphProto* mutable_graph,
                               std::vector<IndexNode> & index_nodes,
                               std::map<std::string, onnx::TensorProto>& weights,
                               std::map<std::string, int>& node_reference,
                               std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // Flatten <= X - Shape - Gather - Constant - Unsqueeze - Unsqueeze - Concat - Reshape
        do {
            if (node->op_type() == "Shape" && i+6 < node_count)
            {
                if (node_reference.find(node->output(0)) == node_reference.end() || node_reference[node->output(0)] != 1)
                    break;

                onnx::NodeProto* node2 = mutable_graph->mutable_node(i+1);
                auto node3 = index_nodes[i+2].node;
                auto node4 = index_nodes[i+3].node;
                auto node5 = index_nodes[i+4].node;
                auto node6 = index_nodes[i+5].node;
                auto node7 = index_nodes[i+6].node;

                if (node2->op_type() != "Gather" ||
                    node3->op_type() != "Constant" ||
                    node4->op_type() != "Unsqueeze" ||
                    node5->op_type() != "Unsqueeze" ||
                    node6->op_type() != "Concat" ||
                    node7->op_type() != "Reshape")
                    break;

                if (node_reference.find(node2->output(0)) == node_reference.end() || node_reference[node2->output(0)] != 1)
                    break;

    //             if (node_reference.find(node3->output(0)) == node_reference.end() || node_reference[node3->output(0)] != 1)
    //                 continue;

                if (node_reference.find(node4->output(0)) == node_reference.end() || node_reference[node4->output(0)] != 1)
                    break;

                if (node_reference.find(node5->output(0)) == node_reference.end() || node_reference[node5->output(0)] != 1)
                    break;

                if (node_reference.find(node6->output(0)) == node_reference.end() || node_reference[node6->output(0)] != 1)
                    break;

                if (node2->input(0) != node->output(0) || node4->input(0) != node2->output(0) || node5->input(0) != node3->output(0)
                    || node6->input(0) != node4->output(0) || node6->input(1) != node5->output(0)
                    || node7->input(0) != node->input(0) || node7->input(1) != node6->output(0))
                    break;

                // axis = 0
                int gather_axis = get_node_attr_i(*node2, "axis");
                if (gather_axis != 0)
                    break;

                // indices = 0
                if (weights.find(node2->input(1)) == weights.end())
                    break;

                std::vector<int64_t> gather_indices = get_tensor_proto_reshape_shape(weights[node2->input(1)]);
                if (gather_indices.size() != 1 || gather_indices[0] != 0)
                    break;

                // axes = (0)
                std::vector<int64_t> unsqueeze_axes = get_node_attr_ai(*node4, "axes");
                if (unsqueeze_axes.size() != 1)
                    break;
                if (unsqueeze_axes[0] != 0)
                    break;

                // axes = (0)
                std::vector<int64_t> unsqueeze2_axes = get_node_attr_ai(*node5, "axes");
                if (unsqueeze2_axes.size() != 1)
                    break;
                if (unsqueeze2_axes[0] != 0)
                    break;

                // data = -1
                if (weights.find(node5->input(0)) == weights.end())
                    break;

                std::vector<int64_t> unsqueeze2_data = get_tensor_proto_reshape_shape(weights[node5->input(0)]);
                if (unsqueeze2_data.size() != 1 || unsqueeze2_data[0] != -1)
                    break;

                // axis = 0
                int concat_axis = get_node_attr_i(*node6, "axis");
                if (concat_axis != 0)
                    break;

                // reduce
                node->set_op_type(k_tnn_noop_type);
                node2->set_op_type(k_tnn_noop_type);
    //             node3->set_op_type(k_tnn_noop_type);
                node4->set_op_type(k_tnn_noop_type);
                node5->set_op_type(k_tnn_noop_type);
                node6->set_op_type(k_tnn_noop_type);

                node_reference[node->input(0)] -= 1;

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node2->output(0)));
    //             node_reference.erase(node_reference.find(node3->output(0)));
                node_reference.erase(node_reference.find(node4->output(0)));
                node_reference.erase(node_reference.find(node5->output(0)));
                node_reference.erase(node_reference.find(node6->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node2->output(0));
    //             blob_names.erase(node3->output(0));
                blob_names.erase(node4->output(0));
                blob_names.erase(node5->output(0));
                blob_names.erase(node6->output(0));

                node7->set_op_type("Flatten");
                node7->clear_input();
                node7->add_input(node->input(0));

                i += 5;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
