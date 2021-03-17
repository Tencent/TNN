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

int Onnx2TNN::TransferGridSample(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                                    std::map<std::string, onnx::TensorProto>& weights,
                                    std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // ReduceMax <= Aten
        do {
            if (node->op_type() == "ATen") {
              auto op = get_node_attr_s(*node, "operator");
              if(op == "grid_sampler") {
                  node->set_op_type("GridSample");
              }
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
