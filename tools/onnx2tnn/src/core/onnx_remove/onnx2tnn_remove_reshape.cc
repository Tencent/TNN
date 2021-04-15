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

int Onnx2TNN::RemoveReshape(onnx::GraphProto* mutable_graph,
                                 std::vector<IndexNode> & index_nodes,
                                 std::map<std::string, onnx::TensorProto>& weights,
                                 std::map<std::string, int>& node_reference,
                                 std::set<std::string>& blob_names) {
    const int node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        onnx::NodeProto* node = index_nodes[i].node;

        //x <= x - Reshape(1, -1 )
        do {
            if (i + 1 >= node_count) {
                break;
            }
            auto node_reshape = index_nodes[i + 1].node;
            if (node_reshape->op_type() != "Reshape")
                break;

            std::vector<int64_t> shape;
            if (node_reshape->input_size() == 1) {
                shape = get_node_attr_ai(*node_reshape, "shape");
            }  else {
                shape = get_tensor_proto_reshape_shape(weights[node_reshape->input(1)]);
            }

//            if (node_reference.find(node->output(0)) == node_reference.end() || node_reference[node->output(0)] != 1)
//                break;

            bool need_remove = false;
            if (shape.size() == 2 && shape[0] == 1 && shape[1] == -1) {
                need_remove = true;
            }

            if (!need_remove) {
                break;
            }
            // reduce
            node_reshape->set_op_type(k_tnn_noop_type);

            auto item = node_reference.find(node_reshape->output(0));
            if (item != node_reference.end()) {
                node_reference.erase(item);
            }
            blob_names.erase(node->output(0));

//            node->set_output(0, node_reshape->output(0));
            RemoveIndexNode(index_nodes, i+1);
        } while (0);

        //x <= x - Reshape(1, n, 1, 1)
        do {
            if (i + 1 >= node_count) {
                break;
            }
            auto node_reshape = index_nodes[i + 1].node;
            if (node_reshape->op_type() != "Reshape")
                break;

            std::vector<int64_t> shape;
            if (node_reshape->input_size() == 1) {
                shape = get_node_attr_ai(*node_reshape, "shape");
            }  else {
                shape = get_tensor_proto_reshape_shape(weights[node_reshape->input(1)]);
            }

//            if (node_reference.find(node->output(0)) == node_reference.end() || node_reference[node->output(0)] != 1)
//                break;

            bool need_remove = false;
            if (shape.size() == 4 && shape[0] == 1 && shape[2] == 1 && shape[3] == 1) {
                need_remove = true;
            }

            if (!need_remove) {
                break;
            }
            // reduce
            node_reshape->set_op_type(k_tnn_noop_type);

            auto item = node_reference.find(node_reshape->output(0));
            if (item != node_reference.end()) {
                node_reference.erase(item);
            }
            blob_names.erase(node->output(0));

//            node->set_output(0, node_reshape->output(0));
            RemoveIndexNode(index_nodes, i+1);
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}


int Onnx2TNN::RemoveConsecutiveReshape(onnx::GraphProto* mutable_graph,
                  std::vector<IndexNode>& index_nodes,
                  std::map<std::string, onnx::TensorProto>& weights,
                  std::map<std::string, int>& node_reference,
                             std::set<std::string>& blob_names) {
    const int node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        onnx::NodeProto* node = index_nodes[i].node;

        //Reshape_1 <= Reshape_0 - Reshape_1
        do {
            if (i + 1 >= node_count) {
                break;
            }
            auto node_reshape_0 = node;
            std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
            if (next_indexes.size() != 1) {
                break;
            }
            auto node_reshape_1 = index_nodes[next_indexes[0]].node;
            
            if (node_reshape_0->op_type() != "Reshape" || node_reshape_1->op_type() != "Reshape")
                break;
            
            //确保两个reshape前后相接，且shape是常量
            if (node_reshape_0->input_size() > 1 && weights.find(node_reshape_0->input(1)) == weights.end()) {
                LOGE("Onnx2TNN::RemoveConsecutiveReshape node_reshape_0 shape is not const\n");
                break;
            }
            
            if (node_reshape_1->input_size() > 1 && weights.find(node_reshape_1->input(1)) == weights.end()) {
                LOGE("Onnx2TNN::RemoveConsecutiveReshape node_reshape_1 shape is not const\n");
                break;
            }

            node_reshape_0->set_op_type(k_tnn_noop_type);

            auto item = node_reference.find(node_reshape_0->output(0));
            if (item != node_reference.end()) {
                node_reference.erase(item);
            }
            blob_names.erase(node_reshape_0->output(0));
            RemoveIndexNode(index_nodes, i);
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
