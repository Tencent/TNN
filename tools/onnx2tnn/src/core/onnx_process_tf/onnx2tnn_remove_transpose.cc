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
#include "onnx_utility.h"

/**
 * 该方法是为了消除 TF 转换为 onnx 后, onnx 模型中有大量的 transpose 操作,
 * 造成某些模型无法转换,并且会影响运行效率.
 * */
int Onnx2TNN::RemoveTranspose(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                              std::map<std::string, onnx::TensorProto>& weights,
                              std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // x <= x - Transpose(0, 2, 3, 1)
        // x <= x - Transpose(0, 3, 1, 2)
        do {
            if (i + 1 >= node_count) {
                break;
            }
            auto node_transpose = index_nodes[i + 1].node;
            if (node_transpose->op_type() != "Transpose")
                break;
            auto perm = get_node_attr_ai(*node_transpose, "perm");
            if (!((perm[0] == 0 && perm[1] == 2 && perm[2] == 3 && perm[3] == 1) ||
                  (perm[0] == 0 && perm[1] == 3 && perm[2] == 1 && perm[3] == 2))) {
                break;
            }

            if (node_reference.find(node_transpose->output(0)) == node_reference.end() ||
                node_reference[node_transpose->output(0)] != 1) {
                break;
            }

            // reduce
            node_transpose->set_op_type(k_tnn_noop_type);

            node_reference.erase(node_reference.find(node_transpose->output(0)));
            blob_names.erase(node_transpose->output(0));

            RemoveIndexNode(index_nodes, i + 1);

        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
