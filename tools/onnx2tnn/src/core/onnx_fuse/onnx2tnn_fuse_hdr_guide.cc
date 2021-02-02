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

int Onnx2TNN::FuseHDRGuide(onnx::GraphProto* mutable_graph,
                                std::vector<IndexNode> & index_nodes,
                                std::map<std::string, onnx::TensorProto>& weights,
                                std::map<std::string, int>& node_reference,
                                std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // HDRGuide <= Conv - Unsqueeze - Sub - Relu - Mul - ReduceSum - Conv - Clip - Squeeze
        do {
            if (node->op_type() == "Conv" && i + 8 < node_count) {
                auto node_conv1 = node;
                auto node_unsqueeze = index_nodes[i+1].node;
                auto node_sub = index_nodes[i+2].node;
                auto node_relu = index_nodes[i+3].node;
                auto node_mul = index_nodes[i+4].node;
                auto node_reduce_sum = index_nodes[i+5].node;
                auto node_conv2 = index_nodes[i+6].node;
                auto node_clip = index_nodes[i+7].node;
                auto node_squeeze = index_nodes[i+8].node;

                // check op
                if (!((node_unsqueeze->op_type() == "Unsqueeze" || node_unsqueeze->op_type() == k_tnn_noop_type) &&
                      node_sub->op_type() == "Sub" &&
                      node_relu->op_type() == "Relu" &&
                      node_mul->op_type() == "Mul" &&
                      node_reduce_sum->op_type() == "ReduceSum" &&
                      node_conv2->op_type() == "Conv" &&
                      node_clip->op_type() == "Clip" &&
                      (node_squeeze->op_type() == "Squeeze" || node_squeeze->op_type() == k_tnn_noop_type))) {
                    break;
                }

                node_unsqueeze->set_op_type(k_tnn_noop_type);
                node_sub->set_op_type(k_tnn_noop_type);
                node_relu->set_op_type(k_tnn_noop_type);
                node_mul->set_op_type(k_tnn_noop_type);
                node_reduce_sum->set_op_type(k_tnn_noop_type);
                node_conv2->set_op_type(k_tnn_noop_type);
                node_clip->set_op_type(k_tnn_noop_type);
                node_squeeze->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_unsqueeze->output(0));
                node_reference.erase(node_sub->output(0));
                node_reference.erase(node_relu->output(0));
                node_reference.erase(node_mul->output(0));
                node_reference.erase(node_reduce_sum->output(0));
                node_reference.erase(node_conv2->output(0));
                node_reference.erase(node_clip->output(0));
                node_reference.erase(node_squeeze->output(0));

                blob_names.erase(node_unsqueeze->output(0));
                blob_names.erase(node_sub->output(0));
                blob_names.erase(node_relu->output(0));
                blob_names.erase(node_mul->output(0));
                blob_names.erase(node_reduce_sum->output(0));
                blob_names.erase(node_conv2->output(0));
                blob_names.erase(node_clip->output(0));
                blob_names.erase(node_squeeze->output(0));

                node_conv1->set_op_type("HDRGuide");
                node_conv1->set_output(0, node_squeeze->output(0));

                //node_sub
                node_conv1->add_input(node_sub->input(1));

                //node_mul
                node_conv1->add_input(node_mul->input(0));

                //node_conv2
                node_conv1->add_input(node_conv2->input(1));
                node_conv1->add_input(node_conv2->input(2));

                i += 8;
            }
        } while (0);

        // HDRGuide <= Conv - Sub - Relu - Mul - ReduceSum - Conv - Clip - Squeeze(0)
        do {
            if (node->op_type() == "Conv" && i + 6 < node_count) {
                int node_index = i+1;
                auto node_conv1 = node;
                auto node_sub = index_nodes[node_index++].node;
                auto node_relu = index_nodes[node_index++].node;
                auto node_mul = index_nodes[node_index++].node;
                auto node_reduce_sum = index_nodes[node_index++].node;
                auto node_conv2 = index_nodes[node_index++].node;
                auto node_clip = index_nodes[node_index++].node;
                auto node_squeeze = node_clip;
                if (node_index < node_count) {
                    node_squeeze = index_nodes[node_index++].node;
                    if (!(node_squeeze->op_type() == "Squeeze" || node_squeeze->op_type() == k_tnn_noop_type)) {
                        node_squeeze = node_clip;
                        node_index--;
                    }
                }

                // check op
                if (!(node_sub->op_type() == "Sub" &&
                      node_relu->op_type() == "Relu" &&
                      node_mul->op_type() == "Mul" &&
                      node_reduce_sum->op_type() == "ReduceSum" &&
                      node_conv2->op_type() == "Conv" &&
                      node_clip->op_type() == "Clip")) {
                    break;
                }

                node_sub->set_op_type(k_tnn_noop_type);
                node_relu->set_op_type(k_tnn_noop_type);
                node_mul->set_op_type(k_tnn_noop_type);
                node_reduce_sum->set_op_type(k_tnn_noop_type);
                node_conv2->set_op_type(k_tnn_noop_type);
                node_clip->set_op_type(k_tnn_noop_type);
                node_squeeze->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_sub->output(0));
                node_reference.erase(node_relu->output(0));
                node_reference.erase(node_mul->output(0));
                node_reference.erase(node_reduce_sum->output(0));
                node_reference.erase(node_conv2->output(0));
                node_reference.erase(node_clip->output(0));
                node_reference.erase(node_squeeze->output(0));

                blob_names.erase(node_sub->output(0));
                blob_names.erase(node_relu->output(0));
                blob_names.erase(node_mul->output(0));
                blob_names.erase(node_reduce_sum->output(0));
                blob_names.erase(node_conv2->output(0));
                blob_names.erase(node_clip->output(0));
                blob_names.erase(node_squeeze->output(0));

                node_conv1->set_op_type("HDRGuide");
                node_conv1->set_output(0, node_squeeze->output(0));

                //node_sub
                node_conv1->add_input(node_sub->input(1));

                //node_mul
                node_conv1->add_input(node_mul->input(0));

                //node_conv2
                node_conv1->add_input(node_conv2->input(1));
                node_conv1->add_input(node_conv2->input(2));

                i = node_index - 1;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
