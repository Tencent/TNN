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

int Onnx2TNN::FuseDepthToSpace(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                               std::map<std::string, onnx::TensorProto>& weights,
                               std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;
        // DepthToSpace <= Reshape - Transpose - Reshape
        // CRD mode: Reshape - Transpose(0, 1, 4, 2, 5, 3) - Reshape
        // DCR mode: Reshape - Transpose(0, 3, 4, 1, 5, 2) - Reshape
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
            if (permute.size() != 6) {
                break;
            }
            std::vector<int64_t> CRD_mode = {0, 1, 4, 2, 5, 3};
            std::vector<int64_t> DCR_mode = {0, 3, 4, 1, 5, 2};
            bool is_crd                   = true;
            bool is_dcr                   = true;
            for (int index = 0; index < 6; index++) {
                if (CRD_mode[index] != permute[index]) {
                    is_crd = false;
                    break;
                }
            }
            for (int index = 0; index < 6; index++) {
                if (DCR_mode[index] != permute[index]) {
                    is_dcr = false;
                    break;
                }
            }
            if (!is_crd && !is_dcr) {
                break;
            }
            if (is_crd && shapes[2] != shapes[3]) {
                break;
            }
            if (is_dcr && shapes[1] != shapes[2]) {
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
            reshape_node->set_op_type("DepthToSpace");
            // reshape_node->clear_input();
            // set input
            // reshape_node->add_input(0, node->input(0));
            reshape_node->set_input(0, node->input(0));
            onnx::AttributeProto* attr_block_size = reshape_node->add_attribute();
            // CRD 和 DCR 中，shapes[2] 都是 block size
            int block_size = shapes[2];
            attr_block_size->set_name("blocksize");
            attr_block_size->set_i(block_size);

            onnx::AttributeProto* attr_mode = reshape_node->add_attribute();
            std::string mode                = is_crd ? "CRD" : "DCR";
            attr_mode->set_s(mode);

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