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
/**
 *  该方法是为了处理 TF 转换为 onnx 后, onnx 模型中 Concat 的axis 的参数
 * |         |   tf(nhwc) |  tnn(nchw)   |
 * |  axis   |     -1     |     1        |
 * |  axis   |     3      |     1        |
 * */
int Onnx2TNN::TransferConcat(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                            std::map<std::string, onnx::TensorProto>& weights,
                            std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;
        do {
            if (node->op_type() == "Concat") {
                auto axis = get_node_attr_i(*node, "axis", 0);
                if (axis == 3 || axis == -1) {
                    auto attr_axis = node->mutable_attribute(0);
                    attr_axis->set_i(1);
                }
            }

        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
