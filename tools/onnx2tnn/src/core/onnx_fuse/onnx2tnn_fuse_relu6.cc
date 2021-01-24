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

int Onnx2TNN::FuseRelu6(onnx::GraphProto* mutable_graph,
                                  std::vector<IndexNode> & index_nodes,
                                  std::map<std::string, onnx::TensorProto>& weights,
                                  std::map<std::string, int>& node_reference,
                                  std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // Relu6 <= Relu - Affine - Affine - Clip - Affine
        do {
            if (node->op_type() == "Relu" && i + 4 < node_count) {
                onnx::NodeProto* node_relu = node;
                onnx::NodeProto* node_affine_0 = index_nodes[i+1].node;
                onnx::NodeProto* node_affine_1 = index_nodes[i+2].node;
                onnx::NodeProto* node_clip = index_nodes[i+3].node;
                onnx::NodeProto* node_affine_2 = index_nodes[i+4].node;
                
                // check op
                if (!(node_affine_0->op_type() == "Affine") ||
                    !(node_affine_1->op_type() == "Affine") ||
                    !(node_clip->op_type() == "Clip") ||
                    !(node_affine_2->op_type() == "Affine"))
                    break;
                
                // check order
                if (node_relu->output(0) != node_affine_0->input(0) ||
                    node_affine_0->output(0) != node_affine_1->input(0) ||
                    node_affine_1->output(0) != node_clip->input(0) ||
                    node_clip->output(0) != node_affine_2->input(0)) {
                    break;
                }
                
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }
                next_indexes = GetNextIndexNode(index_nodes, i+1);
                if (next_indexes.size() != 1) {
                    break;
                }
                next_indexes = GetNextIndexNode(index_nodes, i+2);
                if (next_indexes.size() != 1) {
                    break;
                }
                next_indexes = GetNextIndexNode(index_nodes, i+3);
                if (next_indexes.size() != 1) {
                    break;
                }
                
                float affine_alpha = get_node_attr_f(*node_affine_0, "alpha", onnx_net_info_, 1, 0);
                float affine_beta = get_node_attr_f(*node_affine_0, "beta", onnx_net_info_, 2, 0);
                if (affine_alpha != -1.0f || affine_beta!=0.0f)
                        break;
                
                affine_alpha = get_node_attr_f(*node_affine_1, "alpha", onnx_net_info_, 1, 0);
                affine_beta = get_node_attr_f(*node_affine_1, "beta", onnx_net_info_, 2, 0);
                if (affine_alpha != 1.0f || affine_beta!=0.0f)
                        break;
                
                float relu6_min = get_node_attr_f(*node_clip, "min", onnx_net_info_,1, -FLT_MAX);
                if (relu6_min != -6.0f)
                        break;
                
                affine_alpha = get_node_attr_f(*node_affine_2, "alpha", onnx_net_info_, 1, 0);
                affine_beta = get_node_attr_f(*node_affine_2, "beta", onnx_net_info_, 2, 0);
                if (affine_alpha != -1.0f || affine_beta!=0.0f)
                        break;
                
                node_relu->set_op_type(k_tnn_noop_type);
                node_affine_0->set_op_type(k_tnn_noop_type);
                node_affine_1->set_op_type(k_tnn_noop_type);
                node_clip->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node_relu->output(0)));
                blob_names.erase(node_relu->output(0));
                node_reference.erase(node_reference.find(node_affine_0->output(0)));
                blob_names.erase(node_affine_0->output(0));
                node_reference.erase(node_reference.find(node_affine_1->output(0)));
                blob_names.erase(node_affine_1->output(0));
                node_reference.erase(node_reference.find(node_clip->output(0)));
                blob_names.erase(node_clip->output(0));
                
                node_affine_2->set_op_type("Clip");
                auto attr_min = node_affine_2->add_attribute();
                attr_min->set_name("min");
                attr_min->set_f(0);
                auto attr_max = node_affine_2->add_attribute();
                attr_max->set_name("max");
                attr_max->set_f(6);
                node_affine_2->set_input(0, node_relu->input(0));

                i += 4;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
