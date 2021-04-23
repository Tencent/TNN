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

int Onnx2TNN::FuseMatMul(onnx::GraphProto* mutable_graph,
                              std::vector<IndexNode> & index_nodes,
                              std::map<std::string, onnx::TensorProto>& weights,
                              std::map<std::string, int>& node_reference,
                              std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // MatMul <= Transpose(weight) - MatMul
         do {
             if (node->op_type() == "Transpose") {
                 // check weight
                 if (weights.find(node->input(0)) == weights.end())
                     break;

                 onnx::TensorProto& B = weights[node->input(0)];
                 if (B.dims_size() != 2)
                     break;

                 if (node_reference.find(node->output(0)) == node_reference.end() ||
                     node_reference[node->output(0)] != 1)
                     break;

                 // perm = (1, 0)
                 std::vector<int64_t> perm = get_node_attr_ai(*node, "perm");
                 if (perm.size() != 2)
                     break;
                 if (perm[0] != 1 || perm[1] != 0)
                     break;

                 if (i + 1 >= node_count)
                     break;

                 auto node2 = index_nodes[i+1].node;

                 if (node2->op_type() != "MatMul")
                     break;
                 std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                 if (next_indexes.size() != 1) {
                     break;
                 }

                 // reduce
                 node->set_op_type(k_tnn_noop_type);

                 node_reference.erase(node_reference.find(node->output(0)));
                 blob_names.erase(node->output(0));

                 node2->set_input(1, node->input(0));

                 // permute weight
                 {
                     auto const h = B.dims(0);
                     auto const w = B.dims(1);

                     std::vector<float> permuted_data;
                     permuted_data.reserve(h * w);
                     const float* bptr = B.has_raw_data()
                                             ? (const float*)B.raw_data().data()
                                             : B.float_data().data();

                     for (int j = 0; j < w; j++) {
                         for (int k = 0; k < h; k++) {
                             float vb = bptr[k * w + j];
                             permuted_data.push_back(vb);
                         }
                     }

                     B.set_dims(0, w);
                     B.set_dims(1, h);

                     if (B.has_raw_data()) {
                         B.set_raw_data(permuted_data.data(),
                                        permuted_data.size() * sizeof(float));
                     } else {
                         for (int j = 0; j < (int)permuted_data.size(); j++)
                             B.set_float_data(j, permuted_data[j]);
                     }
                 }

                 i += 1;
             }
         } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
