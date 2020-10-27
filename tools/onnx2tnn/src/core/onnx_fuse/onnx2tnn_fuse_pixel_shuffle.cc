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

#include "half_utils.h"
#include "objseri.h"
#include "onnx2tnn.h"
#include "onnx_utility.h"

int Onnx2TNN::FusePixelShuffle(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                               std::map<std::string, onnx::TensorProto>& weights,
                               std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;
        // PixelShuffle =
        do {
            if (node->op_type() != "Reshape" || i + 2 >= node_count) {
                break;
            }
            auto shapes = get_node_attr_ai(*node, "shape", weights, 1);
            if (shapes.size() != 6) {
                break;
            }
            auto transpose_node = index_nodes[i + 1].node;
            if (transpose_node->op_type() != "Transpose") {
                break;
            }
            auto permute = get_node_attr_ai(*transpose_node, "perm");
            if (permute.size() != 6 || permute[0] != 0 || permute[1] != 1 || permute[2] != 4 || permute[3] != 2 ||
                permute[4] != 5 || permute[5] != 3) {
                break;
            }
            auto reshape_node = index_nodes[i + 2].node;
            if (reshape_node->op_type() != "Reshape") {
                break;
            }
            if (node->output(0) != transpose_node->input(0) || transpose_node->output(0) != reshape_node->input(0)) {
                break;
            }
            if (node_reference[node->output(0)] != 1 || node_reference[transpose_node->output(0)] != 1) {
                break;
            }
            reshape_node->set_op_type("PixelShuffle");
            // reshape_node->clear_input();
            // set input
            // reshape_node->add_input(0, node->input(0));
            reshape_node->set_input(0, node->input(0));
            onnx::AttributeProto* attribute = reshape_node->add_attribute();
            int upscale_factor              = shapes[2];
            attribute->set_name("upscale_factor");
            attribute->set_i(upscale_factor);

            node->set_op_type(k_tnn_noop_type);
            transpose_node->set_op_type(k_tnn_noop_type);
            node_reference.erase(node_reference.find(node->output(0)));
            RemoveIndexNode(index_nodes, i);
            node_reference.erase(node_reference.find(transpose_node->output(0)));
            RemoveIndexNode(index_nodes, i + 1);
            i = i + 2;
        } while (0);
    }
    ClearEmptyNode(index_nodes);
    return 0;
}