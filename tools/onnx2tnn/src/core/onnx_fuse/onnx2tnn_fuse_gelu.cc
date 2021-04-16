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

#include <algorithm>

#include "onnx2tnn.h"

int Onnx2TNN::FuseGELU(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                            std::map<std::string, onnx::TensorProto>& weights,
                            std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;
        // GELU <= Div - Erf - Add(1) - Mul - Mul(0.5)
        do {
            if (node->op_type() == "Div" && i + 4 < node_count) {
                auto node_erf = index_nodes[i + 1].node;
                auto node_add  = index_nodes[i + 2].node;
                auto node_mul1  = index_nodes[i + 3].node;
                auto node_mul2  = index_nodes[i + 4].node;

                if (node_erf->op_type() != "Erf" || node_add->op_type() != "Add" ||
                    node_mul1->op_type() != "Mul" || node_mul2->op_type() != "Mul")
                    break;
                if (node_erf->input(0) != node->output(0) ||
                    node_add->input(0) != node_erf->output(0) ||
                    node_mul2->input(0) != node_mul1->output(0)) {
                    break;
                }
                
                if (node_mul1->input(0) != node->input(0) ||
                    node_mul1->input(1) != node_add->output(0)) {
                    break;
                }
                
                //Div tensor
                if (weights.find(node->input(1)) == weights.end()) {
                    break;
                }
                float div = *get_tensor_proto_data(weights[node->input(1)]);
                if (fabs(div-1.4142135381698608f) > FLT_EPSILON) {
                    break;
                }
                
                //Add tensor
                if (weights.find(node_add->input(1)) == weights.end()) {
                    break;
                }
                float add = *get_tensor_proto_data(weights[node_add->input(1)]);
                if (fabs(add-1.0f) > FLT_EPSILON) {
                    break;
                }
                
                //Mul tensor
                if (weights.find(node_mul2->input(1)) == weights.end()) {
                    break;
                }
                float mul = *get_tensor_proto_data(weights[node_mul2->input(1)]);
                if (fabs(mul-0.5f) > FLT_EPSILON) {
                    break;
                }
                
                // reduce
                node->set_op_type(k_tnn_noop_type);
                node_erf->set_op_type(k_tnn_noop_type);
                node_add->set_op_type(k_tnn_noop_type);
                node_mul1->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node_erf->output(0)));
                node_reference.erase(node_reference.find(node_add->output(0)));
                node_reference.erase(node_reference.find(node_mul1->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node_erf->output(0));
                blob_names.erase(node_add->output(0));
                blob_names.erase(node_mul1->output(0));

                node_mul2->set_op_type("GELU");
                node_mul2->clear_input();
                node_mul2->add_input(node->input(0));
                
                // approximation
                onnx::AttributeProto* attr_approximation  = node_mul2->add_attribute();
                attr_approximation->set_name("approximation");
                attr_approximation->set_i(0);

                i += 4;
            }
        } while (0);
        
        // approximation GELU <= Pow - Mul - Add - Mul - Tanh - Add - Mul(0.5) - Mul
        // the approximation has big error if input is -2.281006575
        do {
            if (node->op_type() == "Pow" && i + 7 < node_count) {
                auto node1 = index_nodes[i + 1].node;
                auto node2  = index_nodes[i + 2].node;
                auto node3  = index_nodes[i + 3].node;
                auto node4  = index_nodes[i + 4].node;
                auto node5  = index_nodes[i + 5].node;
                auto node6  = index_nodes[i + 6].node;
                auto node7  = index_nodes[i + 7].node;

                if (node1->op_type() != "Mul" || node2->op_type() != "Add" ||
                    node3->op_type() != "Mul" || node4->op_type() != "Tanh" ||
                    node5->op_type() != "Add" || node6->op_type() != "Mul" ||
                    node7->op_type() != "Mul")
                    break;
                
                if ((node1->input(0) != node->output(0) && node1->input(1) != node->output(0)) ||
                    (node3->input(0) != node2->output(0) && node3->input(1) != node2->output(0)) ||
                    node4->input(0) != node3->output(0) ||
                    (node5->input(0) != node4->output(0) && node5->input(1) != node4->output(0)) ||
                    (node6->input(0) != node5->output(0) && node6->input(1) != node5->output(0))) {
                    break;
                }
                
                if (node2->input(0) != node->input(0) ||
                    node2->input(1) != node1->output(0)) {
                    break;
                }
                
                if (node7->input(0) != node->input(0) ||
                    node7->input(1) != node6->output(0)) {
                    break;
                }
                
                //TODO check input pow Y = 3
                
                //Mul0 tensor
                if (weights.find(node1->input(0)) == weights.end()) {
                    break;
                }
                float mul0 = *get_tensor_proto_data(weights[node1->input(0)]);
                if (fabs(mul0-0.044714998453855515f) > FLT_EPSILON) {
                    break;
                }
                
                //Add tensor
                if (weights.find(node5->input(0)) == weights.end()) {
                    break;
                }
                float add = *get_tensor_proto_data(weights[node5->input(0)]);
                if (fabs(add-1.0f) > FLT_EPSILON) {
                    break;
                }
                
                //Mul tensor
                if (weights.find(node6->input(0)) == weights.end()) {
                    break;
                }
                float mul = *get_tensor_proto_data(weights[node6->input(0)]);
                if (fabs(mul-0.5f) > FLT_EPSILON) {
                    break;
                }
                
                // reduce
                node->set_op_type(k_tnn_noop_type);
                node1->set_op_type(k_tnn_noop_type);
                node2->set_op_type(k_tnn_noop_type);
                node3->set_op_type(k_tnn_noop_type);
                node4->set_op_type(k_tnn_noop_type);
                node5->set_op_type(k_tnn_noop_type);
                node6->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node1->output(0)));
                node_reference.erase(node_reference.find(node2->output(0)));
                node_reference.erase(node_reference.find(node3->output(0)));
                node_reference.erase(node_reference.find(node4->output(0)));
                node_reference.erase(node_reference.find(node5->output(0)));
                node_reference.erase(node_reference.find(node6->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node1->output(0));
                blob_names.erase(node2->output(0));
                blob_names.erase(node3->output(0));
                blob_names.erase(node4->output(0));
                blob_names.erase(node5->output(0));
                blob_names.erase(node6->output(0));

                node7->set_op_type("GELU");
                node7->clear_input();
                node7->add_input(node->input(0));
                
                // approximation
                onnx::AttributeProto* attr_approximation  = node7->add_attribute();
                attr_approximation->set_name("approximation");
                attr_approximation->set_i(1);

                i += 7;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
