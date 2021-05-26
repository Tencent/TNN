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

int Onnx2TNN::FuseClip(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                       std::map<std::string, onnx::TensorProto>& weights, std::map<std::string, int>& node_reference,
                       std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // Clip <= Cast - Cast - Clip
        do {
            if (node->op_type() == "Cast" && i + 3 < node_count) {
                onnx::NodeProto* node_cast_0 = node;
                onnx::NodeProto* node_cast_1 = index_nodes[i + 1].node;
                onnx::NodeProto* node_clip   = index_nodes[i + 2].node;

                // check op
                if (!(node_cast_1->op_type() == "Cast") || !(node_clip->op_type() == "Clip"))
                    break;

                // check order
                if (node_cast_0->output(0) != node_clip->input(1) || node_cast_1->output(0) != node_clip->input(2)) {
                    break;
                }

                const int to_0 = get_node_attr_i(*node_cast_0, "to", 1);
                const int to_1 = get_node_attr_i(*node_cast_1, "to", 1);
                if (to_0 != 1 || to_1 != 1) {
                    break;
                }

                auto min_val_tensor = weights[node_cast_0->input(0)];
                auto max_val_tensor = weights[node_cast_1->input(0)];
                auto* min_val_ptr   = get_tensor_proto_mutable_data(min_val_tensor);
                auto* max_val_ptr   = get_tensor_proto_mutable_data(max_val_tensor);
                double min_val;
                double max_val;
                if (min_val_tensor.data_type() == onnx::TensorProto_DataType_DOUBLE) {
                    min_val = reinterpret_cast<double*>(min_val_ptr)[0];
                } else {
                    min_val = min_val_ptr[0];
                }
                if (max_val_tensor.data_type() == onnx::TensorProto_DataType_DOUBLE) {
                    max_val = reinterpret_cast<double*>(max_val_ptr)[0];
                } else {
                    max_val = max_val_ptr[0];
                }

                auto input_name = node_clip->input(0);
                node_clip->clear_input();
                node_clip->add_input(input_name);

                onnx::AttributeProto* attr_min = node_clip->add_attribute();
                attr_min->set_name("min");
                attr_min->set_f(min_val);

                onnx::AttributeProto* attr_max = node_clip->add_attribute();
                attr_max->set_name("max");
                attr_max->set_f(max_val);

                node_cast_0->set_op_type(k_tnn_noop_type);
                node_cast_1->set_op_type(k_tnn_noop_type);
                node_reference.erase(node_reference.find(node_cast_0->output(0)));
                node_reference.erase(node_reference.find(node_cast_1->output(0)));
                blob_names.erase(node_cast_0->output(0));
                blob_names.erase(node_cast_1->output(0));

                i += 2;
            }
        } while (0);
    }

    //    ClearEmptyNode(index_nodes);

    return 0;
}
