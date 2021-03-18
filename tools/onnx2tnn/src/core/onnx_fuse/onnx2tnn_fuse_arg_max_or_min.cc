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

int Onnx2TNN::FuseArgMaxOrMin(onnx::GraphProto* mutable_graph,
                                  std::vector<IndexNode> & index_nodes,
                                  std::map<std::string, onnx::TensorProto>& weights,
                                  std::map<std::string, int>& node_reference,
                                  std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // ArgMax <= ArgMax - Unsqueeze(axis = 0)
        do {
            if ((node->op_type() == "ArgMax" || node->op_type() == "ArgMin") && i + 1 < node_count) {
                onnx::NodeProto* node_arg = node;
                
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }
                onnx::NodeProto* node_unsuqeeze = index_nodes[next_indexes[0]].node;
                
                // check op
                if (!(node_unsuqeeze->op_type() == "Unsqueeze"))
                    break;
                
                auto keepdims = get_node_attr_i(*node_arg, "keepdims");
                auto axis_arg = get_node_attr_i(*node_arg, "axis", 0);
                auto axes_unsuqeeze = get_node_attr_ai(*node_unsuqeeze, "axes");
                if (keepdims !=0 || axes_unsuqeeze.size() != 1 ||
                    axis_arg != axes_unsuqeeze[0])
                        break;
                
                node_unsuqeeze->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node_arg->output(0)));
                blob_names.erase(node_arg->output(0));
                
                node_arg->set_output(0, node_unsuqeeze->output(0));
                auto attr = node_arg->mutable_attribute(1);
                attr->set_i(1);

                i += 1;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
