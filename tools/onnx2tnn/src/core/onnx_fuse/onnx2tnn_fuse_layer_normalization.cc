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

#include "onnx2tnn.h"

int Onnx2TNN::FuseLayerNormalization(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                                        std::map<std::string, onnx::TensorProto>& weights,
                                        std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    //NOTE: The fuse logic of layer norm is similar with instance norm. The major difference is the process of scale and bias
    //Unlike Batch Normalization and Instance Normalization, which applies scalar scale and bias for each entire channel/plane with the affine option,
    //Layer Normalization applies per-element scale and bias with elementwise_affine.
    //see https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html?highlight=layernorm#torch.nn.LayerNorm
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;
        /**
         * Fuse for a special model
         * LayerNormalization <= ReduceMean + Sub + Pow + ReduceMean + Add + Sqrt + Div + Mul + Add
         *
         * */
        do {
            string reduce_type = node->op_type();
            if (reduce_type == "ReduceMean" && i + 8 < node_count) {
                auto node1  = index_nodes[i + 1].node;
                auto node2  = index_nodes[i + 2].node;
                auto node3  = index_nodes[i + 3].node;
                auto node4  = index_nodes[i + 4].node;
                auto node5  = index_nodes[i + 5].node;
                auto node6  = index_nodes[i + 6].node;
                auto node7  = index_nodes[i + 7].node;
                auto node8  = index_nodes[i + 8].node;

                if (node1->op_type() != "Sub" || node2->op_type() != "Pow" || node3->op_type() != "ReduceMean" ||
                    node4->op_type() != "Add" || node5->op_type() != "Sqrt" || node6->op_type() != "Div" ||
                    node7->op_type() != "Mul" || node8->op_type() != "Add") {
                    break;
                }
                
                if (node1->input_size() != 2 || node1->input(0) != node->input(0) || node1->input(1) != node->output(0)) {
                    break;
                }

                if (node2->input(0) != node1->output(0) || node3->input(0) != node2->output(0) ||
                    node4->input(0) != node3->output(0) || node5->input(0) != node4->output(0)) {
                    break;
                }
                
                if (node6->input_size() != 2 || node6->input(0) != node1->output(0) || node6->input(1) != node5->output(0)) {
                    break;
                }
                
                if (node7->input(0) != node6->output(0) || node8->input(0) != node7->output(0)) {
                    break;
                }
                
                //scale tensor
                if (weights.find(node7->input(1)) == weights.end()) {
                    break;
                }
                auto tensor_scale = weights.at(node7->input(1));
                auto dims_scale = GetDimsFromTensor(tensor_scale);
                
                //bias tensor
                if (weights.find(node8->input(1)) == weights.end()) {
                    break;
                }
                
                auto axis0 = get_node_attr_ai(*node, "axes");
                auto axis3 = get_node_attr_ai(*node3, "axes");
                auto tensor_bias = weights.at(node8->input(1));
                auto dims_bias = GetDimsFromTensor(tensor_bias);
                
                //check dims
                if (axis0.size() != axis3.size() || axis0.size() != dims_scale.size() ||
                    dims_scale.size() != dims_bias.size()) {
                    break;
                }
                node8->set_op_type("LayerNormalization");
                auto bias_name = node8->input(1);
                node8->clear_input();
                // input
                node8->add_input(node->input(0));
                // Scale
                node8->add_input(node7->input(1));
                // Bias
                node8->add_input(bias_name);
                // epsilon
                const onnx::TensorProto& tensorProto = weights.at(node4->input(1));
                float epsilon = *get_tensor_proto_data(tensorProto);
                onnx::AttributeProto* attr_epsilon   = node8->add_attribute();
                attr_epsilon->set_name("epsilon");
                attr_epsilon->set_f(epsilon);
                
                // axes_size
                onnx::AttributeProto* attr_axis  = node8->add_attribute();
                attr_axis->set_name("reduce_axes_size");
                attr_axis->set_i(axis0.size());

                node->set_op_type(k_tnn_noop_type);
                node1->set_op_type(k_tnn_noop_type);
                node2->set_op_type(k_tnn_noop_type);
                node3->set_op_type(k_tnn_noop_type);
                node4->set_op_type(k_tnn_noop_type);
                node5->set_op_type(k_tnn_noop_type);
                node6->set_op_type(k_tnn_noop_type);
                node7->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node1->output(0)));
                node_reference.erase(node_reference.find(node2->output(0)));
                node_reference.erase(node_reference.find(node3->output(0)));
                node_reference.erase(node_reference.find(node4->output(0)));
                node_reference.erase(node_reference.find(node5->output(0)));
                node_reference.erase(node_reference.find(node6->output(0)));
                node_reference.erase(node_reference.find(node7->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node1->output(0));
                blob_names.erase(node2->output(0));
                blob_names.erase(node3->output(0));
                blob_names.erase(node4->output(0));
                blob_names.erase(node5->output(0));
                blob_names.erase(node6->output(0));
                blob_names.erase(node7->output(0));

                i += 8;
            }
        } while (0);
        
        /**
         * Fuse for a special model
         * LayerNormalization <= ReduceMean + Sub + Cast + Pow + ReduceMean + Add + Sqrt + Div + Mul + Add
         *
         * */
        do {
            string reduce_type = node->op_type();
            if (reduce_type == "ReduceMean" && i + 9 < node_count) {
                auto node1  = index_nodes[i + 1].node;
                auto node1_2  = index_nodes[i + 2].node;
                auto node2  = index_nodes[i + 3].node;
                auto node3  = index_nodes[i + 4].node;
                auto node4  = index_nodes[i + 5].node;
                auto node5  = index_nodes[i + 6].node;
                auto node6  = index_nodes[i + 7].node;
                auto node7  = index_nodes[i + 8].node;
                auto node8  = index_nodes[i + 9].node;

                if (node1->op_type() != "Sub" || node1_2->op_type() != "Cast" ||
                    node2->op_type() != "Pow" || node3->op_type() != "ReduceMean" ||
                    node4->op_type() != "Add" || node5->op_type() != "Sqrt" || node6->op_type() != "Div" ||
                    node7->op_type() != "Mul" || node8->op_type() != "Add") {
                    break;
                }
                
                if (node1->input_size() != 2 || node1->input(0) != node->input(0) || node1->input(1) != node->output(0)) {
                    break;
                }

                if (node1_2->input(0) != node1->output(0) ||
                    node2->input(0) != node1_2->output(0) || node3->input(0) != node2->output(0) ||
                    node4->input(0) != node3->output(0) || node5->input(0) != node4->output(0)) {
                    break;
                }
                
                if (node6->input_size() != 2 || node6->input(0) != node1->output(0) || node6->input(1) != node5->output(0)) {
                    break;
                }
                
                if (node7->input(0) != node6->output(0) || node8->input(0) != node7->output(0)) {
                    break;
                }
                
                //scale tensor
                if (weights.find(node7->input(1)) == weights.end()) {
                    break;
                }
                auto tensor_scale = weights.at(node7->input(1));
                auto dims_scale = GetDimsFromTensor(tensor_scale);
                
                //bias tensor
                if (weights.find(node8->input(1)) == weights.end()) {
                    break;
                }
                
                auto axis0 = get_node_attr_ai(*node, "axes");
                auto axis3 = get_node_attr_ai(*node3, "axes");
                auto tensor_bias = weights.at(node8->input(1));
                auto dims_bias = GetDimsFromTensor(tensor_bias);
                
                //check dims
                if (axis0.size() != axis3.size() || axis0.size() != dims_scale.size() ||
                    dims_scale.size() != dims_bias.size()) {
                    break;
                }
                node8->set_op_type("LayerNormalization");
                auto bias_name = node8->input(1);
                node8->clear_input();
                // input
                node8->add_input(node->input(0));
                // Scale
                node8->add_input(node7->input(1));
                // Bias
                node8->add_input(bias_name);
                // epsilon
                const onnx::TensorProto& tensorProto = weights.at(node4->input(1));
                float epsilon = *get_tensor_proto_data(tensorProto);
                onnx::AttributeProto* attr_epsilon   = node8->add_attribute();
                attr_epsilon->set_name("epsilon");
                attr_epsilon->set_f(epsilon);
                
                // axes_size
                onnx::AttributeProto* attr_axis  = node8->add_attribute();
                attr_axis->set_name("reduce_axes_size");
                attr_axis->set_i(axis0.size());

                node->set_op_type(k_tnn_noop_type);
                node1->set_op_type(k_tnn_noop_type);
                node1_2->set_op_type(k_tnn_noop_type);
                node2->set_op_type(k_tnn_noop_type);
                node3->set_op_type(k_tnn_noop_type);
                node4->set_op_type(k_tnn_noop_type);
                node5->set_op_type(k_tnn_noop_type);
                node6->set_op_type(k_tnn_noop_type);
                node7->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node1->output(0)));
                node_reference.erase(node_reference.find(node1_2->output(0)));
                node_reference.erase(node_reference.find(node2->output(0)));
                node_reference.erase(node_reference.find(node3->output(0)));
                node_reference.erase(node_reference.find(node4->output(0)));
                node_reference.erase(node_reference.find(node5->output(0)));
                node_reference.erase(node_reference.find(node6->output(0)));
                node_reference.erase(node_reference.find(node7->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node1->output(0));
                blob_names.erase(node1_2->output(0));
                blob_names.erase(node2->output(0));
                blob_names.erase(node3->output(0));
                blob_names.erase(node4->output(0));
                blob_names.erase(node5->output(0));
                blob_names.erase(node6->output(0));
                blob_names.erase(node7->output(0));

                i += 9;
            }
        } while (0);
        
        /**
         * Fuse for a special model
         * LayerNormalization <= ReduceMean + Sub + Mul + ReduceMean + Add + Sqrt + Reciprocal + Mul + Mul
         *                          + Sub + Add + Mul + Add
         *
         * */
        do {
            string reduce_type = node->op_type();
            if (reduce_type == "ReduceMean" && i + 11 < node_count) {
                auto node1  = index_nodes[i + 1].node;
                auto node2  = index_nodes[i + 2].node;
                auto node3  = index_nodes[i + 3].node;
                auto node4  = index_nodes[i + 4].node;
                auto node5  = index_nodes[i + 5].node;
                auto node6  = index_nodes[i + 6].node;
                auto node7  = index_nodes[i + 7].node;
                auto node8  = index_nodes[i + 8].node;
                auto node9  = index_nodes[i + 9].node;
                auto node10 = index_nodes[i + 10].node;
                auto node11 = index_nodes[i + 11].node;

                if (node1->op_type() != "Sub" || node2->op_type() != "Mul" || node3->op_type() != "ReduceMean" ||
                    node4->op_type() != "Add" || node5->op_type() != "Sqrt" || node6->op_type() != "Reciprocal" ||
                    node7->op_type() != "Mul" || node8->op_type() != "Mul" || node9->op_type() != "Sub" ||
                    node10->op_type() != "Mul" || node11->op_type() != "Add") {
                    break;
                }
                if (i == 0) {
                    break;
                }
                auto node_before = index_nodes[i - 1].node;
                if (node->input(0) != node_before->output(0) || node1->input(0) != node_before->output(0) ||
                    node1->input(1) != node->output(0)) {
                    break;
                }

                if (node2->input(0) != node1->output(0) || node3->input(0) != node2->output(0) ||
                    node4->input(0) != node3->output(0) || node5->input(0) != node4->output(0) ||
                    node6->input(0) != node5->output(0) || node7->input(0) != node6->output(0)) {
                    break;
                }
                // Mul(X, X)
                if (node2->input(0) != node2->input(1)) {
                    break;
                }
                if (!((node8->input(0) == node7->output(0) && node8->input(1) == node->output(0)) ||
                      (node8->input(0) == node->output(0) && node8->input(1) == node7->output(0)))) {
                    break;
                }
                if (node9->input(1) != node8->output(0)) {
                    break;
                }

                if (!((node10->input(0) == node_before->output(0) && node10->input(1) == node7->output(0)) ||
                      (node10->input(0) == node7->output(0) && node10->input(1) == node_before->output(0)))) {
                    break;
                }
                if (!((node11->input(0) == node9->output(0) && node11->input(1) == node10->output(0)) ||
                      (node11->input(0) == node10->output(0) && node11->input(1) == node9->output(0)))) {
                    break;
                }

                //scale tensor
                if (weights.find(node7->input(1)) == weights.end()) {
                    break;
                }
                auto tensor_scale = weights.at(node7->input(1));
                auto dims_scale = GetDimsFromTensor(tensor_scale);
                
                //bias tensor
                if (weights.find(node9->input(0)) == weights.end()) {
                    break;
                }
                
                auto axis0 = get_node_attr_ai(*node, "axes");
                auto axis3 = get_node_attr_ai(*node3, "axes");
                auto tensor_bias = weights.at(node9->input(0));
                auto dims_bias = GetDimsFromTensor(tensor_bias);
                
                //check dims
                if (axis0.size() != axis3.size() || axis0.size() != dims_scale.size() ||
                    dims_scale.size() != dims_bias.size()) {
                    break;
                }
                
                node11->set_op_type("LayerNormalization");
                node11->clear_input();
                // input
                node11->add_input(node->input(0));
                // Scale
                node11->add_input(node7->input(1));
                // Bias
                node11->add_input(node9->input(0));
                // epsilon
                const onnx::TensorProto& tensorProto = weights.at(node4->input(1));
                float epsilon                        = *get_tensor_proto_data(tensorProto);
                onnx::AttributeProto* attr_epsilon   = node11->add_attribute();
                attr_epsilon->set_name("epsilon");
                attr_epsilon->set_f(epsilon);
                
                // axes_size
                onnx::AttributeProto* attr_axis  = node11->add_attribute();
                attr_axis->set_name("reduce_axes_size");
                attr_axis->set_i(axis0.size());

                node->set_op_type(k_tnn_noop_type);
                node1->set_op_type(k_tnn_noop_type);
                node2->set_op_type(k_tnn_noop_type);
                node3->set_op_type(k_tnn_noop_type);
                node4->set_op_type(k_tnn_noop_type);
                node5->set_op_type(k_tnn_noop_type);
                node6->set_op_type(k_tnn_noop_type);
                node7->set_op_type(k_tnn_noop_type);
                node8->set_op_type(k_tnn_noop_type);
                node9->set_op_type(k_tnn_noop_type);
                node10->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node1->output(0)));
                node_reference.erase(node_reference.find(node2->output(0)));
                node_reference.erase(node_reference.find(node3->output(0)));
                node_reference.erase(node_reference.find(node4->output(0)));
                node_reference.erase(node_reference.find(node5->output(0)));
                node_reference.erase(node_reference.find(node6->output(0)));
                node_reference.erase(node_reference.find(node7->output(0)));
                node_reference.erase(node_reference.find(node8->output(0)));
                node_reference.erase(node_reference.find(node9->output(0)));
                node_reference.erase(node_reference.find(node10->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node1->output(0));
                blob_names.erase(node2->output(0));
                blob_names.erase(node3->output(0));
                blob_names.erase(node4->output(0));
                blob_names.erase(node5->output(0));
                blob_names.erase(node6->output(0));
                blob_names.erase(node7->output(0));
                blob_names.erase(node8->output(0));
                blob_names.erase(node9->output(0));
                blob_names.erase(node10->output(0));

                i += 11;
            }
        } while (0);
    }
    ClearEmptyNode(index_nodes);
    return 0;
}
