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

int Onnx2TNN::FuseHistogram(onnx::GraphProto* mutable_graph,
                                  std::vector<IndexNode> & index_nodes,
                                  std::map<std::string, onnx::TensorProto>& weights,
                                  std::map<std::string, int>& node_reference,
                                  std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // Histogram <= OneHot - Cast - MatMul
        do {
            if (node->op_type() == "OneHot" && i + 2 < node_count) {
                onnx::NodeProto* node_onehot = node;
                
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }
                onnx::NodeProto* node_cast = index_nodes[next_indexes[0]].node;
                
                // check op
                if (node_cast->op_type() != "Cast")
                    break;
                auto to = get_node_attr_i(*node_cast, "to", 1);
                if (to != 6)
                        break;
                
                next_indexes = GetNextIndexNode(index_nodes, next_indexes[0]);
                if (next_indexes.size() != 1) {
                    break;
                }
                onnx::NodeProto* node_matmul = index_nodes[next_indexes[0]].node;
                // check op
                if (node_matmul->op_type() != "MatMul")
                    break;
                
                node_onehot->set_op_type("Histogram");
                node_cast->set_op_type(k_tnn_noop_type);
                node_matmul->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node_cast->output(0)));
                blob_names.erase(node_cast->output(0));
                node_reference.erase(node_reference.find(node_matmul->output(0)));
                blob_names.erase(node_matmul->output(0));
                
                node_onehot->set_output(0, node_matmul->output(0));

                i += 2;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
