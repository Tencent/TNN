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

int Onnx2TNN::FuseSwish(onnx::GraphProto* mutable_graph,
                                   std::vector<IndexNode> & index_nodes,
                                   std::map<std::string, onnx::TensorProto>& weights,
                                   std::map<std::string, int>& node_reference,
                                   std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    //for now, we only implement the fusion logic and cpu operator. cancel "return 0" if implementation are all done for device opencl、arm、cuda and metal
    //remember to implement the activation of conv
    //return 0;

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;
        // Swish <= Sigmoid - Mul
        do {
            if (node->op_type() == "Sigmoid" && i+1 < node_count) {
                if (node_reference.find(node->output(0)) == node_reference.end() ||
                    node_reference[node->output(0)] != 1)
                     break;
                 auto node2 = index_nodes[i+1].node;

                 if (node2->op_type() != "Mul" || node2->input_size() != 2)
                     break;

                 if(!((node2->input(0) == node->input(0) && node2->input(1) == node->output(0)) ||
                      (node2->input(1) == node->input(0) && node2->input(0) == node->output(0))))
                     break;

                 // reduce
                 node->set_op_type(k_tnn_noop_type);
                 node2->set_op_type("Swish");
                 node2->clear_input();
                 node2->add_input(node->input(0));

                 node_reference.erase(node_reference.find(node->output(0)));
                 blob_names.erase(node->output(0));

                 i += 1;
             }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
