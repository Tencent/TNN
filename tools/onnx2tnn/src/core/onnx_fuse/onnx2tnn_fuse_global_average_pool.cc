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
#include "onnx_utility.h"

int Onnx2TNN::FuseGlobalAveragePool(
    onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
    std::map<std::string, onnx::TensorProto>& weights,
    std::map<std::string, int>& node_reference,
    std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    // GlobalAveragePool <= Transpose + ReduceMean
    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        do {
            if (node->op_type() == "Transpose" && i + 1 < node_count) {
                auto node_trans = node;

                std::vector<int> next_indexes =
                    GetNextIndexNode(index_nodes, i);
      
                if (next_indexes.size() != 1) {
                    break;
                }

                if (node_trans->output_size() !=1){
                    break;
                }

                auto node_reduce_mean = index_nodes[i + 1].node;

                // check op
                if (!(node_reduce_mean->op_type() == "ReduceMean"))
                    break;
                
                //check keepdim
                auto keepdims = get_node_attr_i(*node_trans, "keepdims", 0);
                if (keepdims  != 1) {
                    break;
                }
                
                // check transpose
                if (node_trans->output(0) != node_reduce_mean->input(0)) {
                    break;
                }
                // check transpose perm
                vector<int64_t> perm = get_node_attr_ai(*node_trans, "perm");
                if (perm.size() != 4 || perm[0] != 0 || perm[1] != 2 ||
                    perm[2] != 3 || perm[3] != 1) {
                    break;
                }
                // check reduce mean axes
                vector<int64_t> axes = get_node_attr_ai(*node_reduce_mean, "axes");
                if (axes.size() != 2 || axes[0] != 1 || axes[1] != 2) {
                    break;
                }


                node_reduce_mean->set_op_type(k_tnn_noop_type);
                node_trans->set_op_type("GlobalAveragePool");
                node_reference.erase(node_reference.find(node_trans->output(0)));
                blob_names.erase(node_trans->output(0));
                node_trans->set_output(0, node_reduce_mean->output(0));

                i += 1;
            }
            if(node->op_type() == "ReduceMean"){
                //check reduce mean axes
                auto node_reduce_mean =node;
                
                //check keepdim
                auto keepdims = get_node_attr_i(*node_reduce_mean, "keepdims", 0);
                if (keepdims  != 1) {
                    break;
                }
                
                vector<int64_t> axes = get_node_attr_ai(*node_reduce_mean, "axes");
                if (axes.size() != 2 || axes[0] != 2 || axes[1] != 3) {
                    break;
                }
                node_reduce_mean->set_op_type("GlobalAveragePool");
                node_reference.erase(node_reference.find(node_reduce_mean->output(0)));
                blob_names.erase(node_reduce_mean->output(0));

            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
