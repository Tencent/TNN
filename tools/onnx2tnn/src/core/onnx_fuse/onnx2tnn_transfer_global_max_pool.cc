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

int Onnx2TNN::TransferGlobalMaxPool(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                                    std::map<std::string, onnx::TensorProto>& weights,
                                    std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    // GlobalMaxPool <=  ReduceMax
    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        do {
            if (node->op_type() == "ReduceMax") {
                // check reduce max axes
                auto node_reduce_max = node;
                vector<int64_t> axes = get_node_attr_ai(*node_reduce_max, "axes");
                if (axes.size() != 2 || axes[0] != 2 || axes[1] != 3) {
                    break;
                }
                node_reduce_max->clear_attribute();
                node_reduce_max->set_op_type("GlobalMaxPool");
                node_reference.erase(node_reference.find(node_reduce_max->output(0)));
                blob_names.erase(node_reduce_max->output(0));
            }
        } while (0);
    }

    return 0;
}
