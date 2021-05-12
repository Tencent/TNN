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
#include <limits.h>

#include "onnx2tnn.h"

int Onnx2TNN::FuseNormalize(onnx::GraphProto* mutable_graph,
                                 std::vector<IndexNode> & index_nodes,
                                 std::map<std::string, onnx::TensorProto>& weights,
                                 std::map<std::string, int>& node_reference,
                                 std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // Normalize <= X - ReduceL2 - Clip - Shape - Expand - Div
        do {
            string reduce_type = node->op_type();
            if ((reduce_type == "ReduceL1" || reduce_type == "ReduceL2" ||
                 reduce_type == "ReduceMax" || reduce_type == "ReduceMin") && i+4 < node_count) {
                if (node_reference.find(node->output(0)) == node_reference.end() ||
                    node_reference[node->output(0)] != 1)
                    break;

                // axes = (1)
                std::vector<int64_t> axes = get_node_attr_ai(*node, "axes");
                if (axes.size() != 1)
                    break;
                if (axes[0] != 1)
                    break;

                auto node2 = index_nodes[i+1].node;
                auto node3 = index_nodes[i+2].node;
                auto node4 = index_nodes[i+3].node;
                auto node5 = index_nodes[i+4].node;

                if (node2->op_type() != "Clip" ||
                    node3->op_type() != "Shape" ||
                    node4->op_type() != "Expand" ||
                    node5->op_type() != "Div")
                    break;

                if (node_reference.find(node2->output(0)) == node_reference.end() ||
                    node_reference[node2->output(0)] != 1)
                    break;

                if (node_reference.find(node3->output(0)) == node_reference.end() ||
                    node_reference[node3->output(0)] != 1)
                    break;

                if (node_reference.find(node4->output(0)) == node_reference.end() ||
                    node_reference[node4->output(0)] != 1)
                    break;

                if (node2->input(0) != node->output(0) || node3->input(0) != node->input(0)
                    || node4->input(0) != node2->output(0) || node4->input(1) != node3->output(0)
                    || node5->input(0) != node->input(0) || node5->input(1) != node4->output(0))
                    break;

                // +eps
                float clip_min = get_node_attr_f(*node2, "min", 0.f);

                // reduce
                node->set_op_type(k_tnn_noop_type);
                node2->set_op_type(k_tnn_noop_type);
                node3->set_op_type(k_tnn_noop_type);
                node4->set_op_type(k_tnn_noop_type);

                node_reference[node->input(0)] -= 2;

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node2->output(0)));
                node_reference.erase(node_reference.find(node3->output(0)));
                node_reference.erase(node_reference.find(node4->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node2->output(0));
                blob_names.erase(node3->output(0));
                blob_names.erase(node4->output(0));

                node5->set_op_type("Normalize");
                node5->clear_input();
                node5->add_input(node->input(0));

                onnx::AttributeProto* attr_eps = node5->add_attribute();
                attr_eps->set_name("eps");
                attr_eps->set_f(clip_min);

                onnx::AttributeProto* attr_dim = node5->add_attribute();
                attr_dim->set_name("dim");
                attr_dim->set_i(1);

                onnx::AttributeProto* attr_p = node5->add_attribute();
                attr_p->set_name("p");
                if (reduce_type == "ReduceL1") { //1范数
                    attr_p->set_i(1);//
                } else if (reduce_type == "ReduceL2") { //2范数
                    attr_p->set_i(2);//
                } else if (reduce_type == "ReduceMax") { //无穷大范数
                    attr_p->set_i(INT_MAX);//
                } else if (reduce_type == "ReduceMin") { //无穷小范数
                    attr_p->set_i(INT_MIN);//
                }

                i += 4;
            }
        } while (0);

        // Normalize <= X - ReduceL2 - Clip - Expand - Div
        do {
            string reduce_type = node->op_type();
            if ((reduce_type == "ReduceL1" || reduce_type == "ReduceL2" ||
                 reduce_type == "ReduceMax" || reduce_type == "ReduceMin") && i+3 < node_count) {
                if (node_reference.find(node->output(0)) == node_reference.end() ||
                    node_reference[node->output(0)] != 1)
                    break;

                // axes = (1)
                std::vector<int64_t> axes = get_node_attr_ai(*node, "axes");
                if (axes.size() != 1)
                    break;
                if (axes[0] != 1)
                    break;

                auto node2 = index_nodes[i+1].node;
                auto node3 = index_nodes[i+2].node;
                auto node4 = index_nodes[i+3].node;

                if (node2->op_type() != "Clip" ||
                    node3->op_type() != "Expand" ||
                    node4->op_type() != "Div")
                    break;

                if (node_reference.find(node2->output(0)) == node_reference.end() ||
                    node_reference[node2->output(0)] != 1)
                    break;

                if (node_reference.find(node3->output(0)) == node_reference.end() ||
                    node_reference[node3->output(0)] != 1)
                    break;

                if (node2->input(0) != node->output(0) || node3->input(0) != node2->output(0)
                    || node4->input(0) != node->input(0) || node4->input(1) != node3->output(0))
                    break;

                // +eps
                float clip_min = get_node_attr_f(*node2, "min", 0.f);

                // reduce
                node->set_op_type(k_tnn_noop_type);
                node2->set_op_type(k_tnn_noop_type);
                node3->set_op_type(k_tnn_noop_type);

                node_reference[node->input(0)] -= 1;

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node2->output(0)));
                node_reference.erase(node_reference.find(node3->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node2->output(0));
                blob_names.erase(node3->output(0));

                node4->set_op_type("Normalize");
                node4->clear_input();
                node4->add_input(node->input(0));

                onnx::AttributeProto* attr_eps = node4->add_attribute();
                attr_eps->set_name("eps");
                attr_eps->set_f(clip_min);

                onnx::AttributeProto* attr_dim = node4->add_attribute();
                attr_dim->set_name("dim");
                attr_dim->set_i(1);

                onnx::AttributeProto* attr_p = node4->add_attribute();
                attr_p->set_name("p");
                if (reduce_type == "ReduceL1") { //1范数
                    attr_p->set_i(1);//
                } else if (reduce_type == "ReduceL2") { //2范数
                    attr_p->set_i(2);//
                } else if (reduce_type == "ReduceMax") { //无穷大范数
                    attr_p->set_i(INT_MAX);//
                } else if (reduce_type == "ReduceMin") { //无穷小范数
                    attr_p->set_i(INT_MIN);//
                }

                i += 3;
            }
        } while (0);

        do {
            // Normalize <= X - Mul(square) - ReduceSum - Max - Sqrt - Reciprocal - Mul
            // node: Mul(X, X)
            string node_type = node->op_type();
            if (node_type != "Mul" || node->input_size() != 2 ||
                node->input(0) != node->input(1) || i + 1 >= node_count) {
                break;
            }
            // node2: Reduce
            auto node2 = index_nodes[i+1].node;
            string reduce_type = node2->op_type();
            if ((reduce_type == "ReduceL1" || reduce_type == "ReduceL2" ||
                reduce_type == "ReduceMax" || reduce_type == "ReduceMin" ||
                reduce_type == "ReduceSum") && i+5 < node_count) {
                if (node_reference.find(node2->output(0)) == node_reference.end() ||
                    node_reference[node2->output(0)] != 1)
                    break;

                // axes = (1)
                std::vector<int64_t> axes = get_node_attr_ai(*node2, "axes");
                if (axes.size() != 1)
                    break;
                if (axes[0] != 1)
                    break;

                auto node3 = index_nodes[i+2].node;
                auto node4 = index_nodes[i+3].node;
                auto node5 = index_nodes[i+4].node;
                auto node6 = index_nodes[i+5].node;

                if (node3->op_type() != "Max" ||
                    node4->op_type() != "Sqrt" ||
                    node5->op_type() != "Reciprocal" ||
                    node6->op_type() != "Mul")
                    break;


                if (node_reference.find(node3->output(0)) == node_reference.end() ||
                    node_reference[node3->output(0)] != 1)
                    break;

                if (node_reference.find(node4->output(0)) == node_reference.end() ||
                    node_reference[node4->output(0)] != 1)
                    break;

                if (node_reference.find(node5->output(0)) == node_reference.end() ||
                    node_reference[node5->output(0)] != 1)
                    break;

                if (node2->input(0) != node->output(0) ||
                    node3->input(0) != node2->output(0) ||
                    node4->input(0) != node3->output(0) ||
                    node5->input(0) != node4->output(0) )
                    break;
                if (node6->input(0) != node5->output(0) &&
                    node6->input(1) != node5->output(0)) {
                    break;
                }
                // +eps
                //float clip_min = get_node_attr_f(*node3, "min", 0.f);
                if (get_tensor_proto_data_size(weights[node3->input(1)]) != 1 ) {
                    break;
                }
                // get float value
                float eps = *(get_tensor_proto_data(weights[node3->input(1)]));

                // reduce
                node->set_op_type(k_tnn_noop_type);
                node2->set_op_type(k_tnn_noop_type);
                node3->set_op_type(k_tnn_noop_type);
                node4->set_op_type(k_tnn_noop_type);
                node5->set_op_type(k_tnn_noop_type);

                node_reference[node->input(0)] -= 1;

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node2->output(0)));
                node_reference.erase(node_reference.find(node3->output(0)));
                node_reference.erase(node_reference.find(node4->output(0)));
                node_reference.erase(node_reference.find(node5->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node2->output(0));
                blob_names.erase(node3->output(0));
                blob_names.erase(node4->output(0));
                blob_names.erase(node5->output(0));

                node6->set_op_type("Normalize");
                node6->clear_input();
                node6->add_input(node->input(0));

                onnx::AttributeProto* attr_eps = node6->add_attribute();
                attr_eps->set_name("eps");
                attr_eps->set_f(eps);

                onnx::AttributeProto* attr_dim = node6->add_attribute();
                attr_dim->set_name("dim");
                attr_dim->set_i(1);

                onnx::AttributeProto* attr_p = node6->add_attribute();
                attr_p->set_name("p");
                if (reduce_type == "ReduceL1") { //1范数
                    attr_p->set_i(1);//
                } else if (reduce_type == "ReduceL2") { //2范数
                    attr_p->set_i(2);//
                } else if (reduce_type == "ReduceMax") { //无穷大范数
                    attr_p->set_i(INT_MAX);//
                } else if (reduce_type == "ReduceMin") { //无穷小范数
                    attr_p->set_i(INT_MIN);//
                } else if (reduce_type == "ReduceSum") {
                    attr_p->set_i(2);
                }

                i += 5;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
