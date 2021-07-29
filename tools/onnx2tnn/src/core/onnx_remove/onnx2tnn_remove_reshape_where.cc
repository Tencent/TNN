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

int Onnx2TNN::RemoveReshapeWhere(onnx::GraphProto* mutable_graph,
                              std::vector<IndexNode> & index_nodes,
                              std::map<std::string, onnx::TensorProto>& weights,
                              std::map<std::string, int>& node_reference,
                              std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        //x <= x - Reshape(-1) - Shape - ConstantOfShape - Mul - Equal - (Cast) - Where
        do {
            std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
            if (next_indexes.size() != 1) {
                break;
            }
            
            onnx::NodeProto *node_x = node;
            
            onnx::NodeProto *node_reshape = nullptr;
            onnx::NodeProto *node_shape = nullptr;
            onnx::NodeProto *node_constantofshape = nullptr;
            onnx::NodeProto *node_mul = nullptr;
            onnx::NodeProto *node_equal = nullptr;
            onnx::NodeProto *node_cast = nullptr;
            onnx::NodeProto *node_where = nullptr;
            int node_where_index = 0;
            
            node_reshape = index_nodes[next_indexes[0]].node;
            if (node_reshape->op_type() != "Reshape")
                break;
            
            {
                next_indexes = GetNextIndexNode(index_nodes, next_indexes[0]);
                if (next_indexes.size() != 3) {
                    break;
                }
                
                node_shape = index_nodes[next_indexes[0]].node;
                node_equal = index_nodes[next_indexes[1]].node;
                node_where = index_nodes[next_indexes[2]].node;
                node_where_index = next_indexes[2];
                
                if (node_shape->op_type() != "Shape" || node_equal->op_type() != "Equal" ||
                    node_where->op_type() != "Where")
                    break;
                
                {
                    auto equal_next_indexes = GetNextIndexNode(index_nodes, next_indexes[1]);
                    if (equal_next_indexes.size() != 1) {
                        break;
                    }
                    node_cast = index_nodes[equal_next_indexes[0]].node;
                    if (node_cast->op_type() != "Cast" && node_cast->op_type() != "Where")
                        break;
                }

            }
            
            {
                next_indexes = GetNextIndexNode(index_nodes, next_indexes[0]);
                if (next_indexes.size() != 1) {
                    break;
                }
                
                node_constantofshape = index_nodes[next_indexes[0]].node;
                if (node_constantofshape->op_type() != "ConstantOfShape")
                    break;
            }

            {
                next_indexes = GetNextIndexNode(index_nodes, next_indexes[0]);
                if (next_indexes.size() != 2) {
                    break;
                }
                
                node_mul = index_nodes[next_indexes[0]].node;
                if (node_mul->op_type() != "Mul")
                    break;
            }
            
            // reduce
            node_reshape->set_op_type(k_tnn_noop_type);
            node_shape->set_op_type(k_tnn_noop_type);
            node_constantofshape->set_op_type(k_tnn_noop_type);
            node_mul->set_op_type(k_tnn_noop_type);
            node_equal->set_op_type(k_tnn_noop_type);
            node_cast->set_op_type(k_tnn_noop_type);
            node_where->set_op_type(k_tnn_noop_type);

            node_reference.erase(node_reference.find(node_x->output(0)));
            blob_names.erase(node_x->output(0));
            node_reference.erase(node_reference.find(node_reshape->output(0)));
            blob_names.erase(node_reshape->output(0));
            node_reference.erase(node_reference.find(node_shape->output(0)));
            blob_names.erase(node_shape->output(0));
            node_reference.erase(node_reference.find(node_constantofshape->output(0)));
            blob_names.erase(node_constantofshape->output(0));
            node_reference.erase(node_reference.find(node_mul->output(0)));
            blob_names.erase(node_mul->output(0));
            node_reference.erase(node_reference.find(node_equal->output(0)));
            blob_names.erase(node_equal->output(0));
            if (node_cast != node_where) {
                node_reference.erase(node_reference.find(node_cast->output(0)));
                blob_names.erase(node_cast->output(0));
            }

            {
                auto node_where_output = node_where->output(0);
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, node_where_index);
                for (auto index : next_indexes) {
                    auto next_node = index_nodes[index].node;
                    for (int ii = 0; ii < next_node->input_size(); ii++) {
                        if (node_where_output == next_node->input(ii)) {
                            next_node->set_input(ii, node_x->output(0));
                        }
                    }
                }
            }

        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
