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

bool IsTensorProtoZero(const onnx::TensorProto& tp) {
    if (tp.has_raw_data()) {
        for (auto val : tp.raw_data()) {
            if (val != 0) {
                return false;
            }
        }
    } else if (tp.data_type() == 1) {
        for (auto val : tp.float_data()) {
            if (val != 0) {
                return false;
            }
        }
    } else if (tp.data_type() == 6) {
        for (auto val : tp.int32_data()) {
            if (val != 0) {
                return false;
            }
        }
    } else if (tp.data_type() == 7) {
        for (auto val : tp.int64_data()) {
            if (val != 0) {
                return false;
            }
        }
    } else if (tp.data_type() == 11) {
        for (auto val : tp.double_data()) {
            if (val != 0) {
                return false;
            }
        }
    } else {
        printf("name:%s data_type :%d\n", tp.name().c_str(), tp.data_type());
        assert(0);
        return false;
    }
    return true;
}

int Onnx2TNN::FuseScatterElements(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                                  std::map<std::string, onnx::TensorProto>& weights,
                                  std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        do {
            if ((node->op_type() == "ScatterElements") && i + 1 < node_count) {
                onnx::NodeProto* node_cur = node;

                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }
                onnx::NodeProto* node_next = index_nodes[next_indexes[0]].node;

                // check op
                if (!(node_next->op_type() == "Add"))
                    break;

                // checke data in ScatterElements
                if (weights.find(node_cur->input(0)) == weights.end())
                    break;
                auto data_tensor = weights[node_cur->input(0)];
                if (!IsTensorProtoZero(data_tensor))
                    break;

                // add op attribute
                onnx::AttributeProto* attr_op = node_cur->add_attribute();
                attr_op->set_name("op");
                attr_op->set_i(1);

                // change input and output
                node_cur->set_input(0, node_next->input(0));
                node_cur->set_output(0, node_next->output(0));
                node_next->set_op_type(k_tnn_noop_type);

                i += 1;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
