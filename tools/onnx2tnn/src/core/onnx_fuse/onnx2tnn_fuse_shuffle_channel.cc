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

int Onnx2TNN::FuseShuffleChannel(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                                 std::map<std::string, onnx::TensorProto>& weights,
                                 std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // ShuffleChannel <= Reshape - Transpose - Reshape
        do {
            if (node->op_type() == "Reshape" && i + 2 < node_count) {
                if (node_reference[node->output(0)] != 1) {
                    break;
                }

                auto node2 = index_nodes[i + 1].node;
                auto node3 = index_nodes[i + 2].node;
                if (node2->op_type() != "Transpose" || node3->op_type() != "Reshape") {
                    break;
                }

                if (node_reference[node2->output(0)] != 1) {
                    break;
                }

                std::vector<int64_t> shape1 = get_node_attr_ai(*node, "shape", onnx_net_info_, 1);
                std::vector<int64_t> perm   = get_node_attr_ai(*node2, "perm");
                std::vector<int64_t> shape3 = get_node_attr_ai(*node3, "shape", onnx_net_info_, 1);

                int64_t group = 0;

                if (shape1.size() == 5 && perm.size() == 5) {
                    // batch groups channels_per_group, height, width
                    group = shape1[1];

                    // 0 2 1 3 4
                    if (perm[0] != 0 || perm[1] != 2 || perm[2] != 1 || perm[3] != 3 || perm[4] != 4) {
                        break;
                    }

                    if (shape3.size() != 4 || shape3[0] != shape1[0] ||
                        (shape3[1] != -1 && shape3[1] != shape1[1] * shape1[2]) || shape3[2] != shape1[3] ||
                        shape3[3] != shape1[4]) {
                        break;
                    }
                } else if (shape1.size() == 3 && perm.size() == 3) {
                    // groups, channels_per_group, height*width
                    group = shape1[0];
                    // 1 0 2
                    if (perm[0] != 1 || perm[1] != 0 || perm[2] != 2) {
                        break;
                    }

                    // TODO：考虑情况shape3各种大小
                    if (shape3.size() != 5 || shape3[0] != shape1[0] ||
                        (shape3[1] != -1 && shape3[1] != shape1[1] * shape1[2]) || shape3[2] != shape1[3]) {
                        break;
                    }
                } else {
                    break;
                }
                // or batch groups channels_per_group, height*width

                // reduce
                node->set_op_type(k_tnn_noop_type);
                node2->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node2->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node2->output(0));

                node3->set_op_type("ShuffleChannel");
                node3->set_input(0, node->input(0));

                onnx::AttributeProto* attr_group = node3->add_attribute();
                attr_group->set_name("group");
                attr_group->set_i(group);

                i += 2;
            }
        } while (0);

        // ShuffleChannel - StrideSlice - StrideSlice <= Reshape - Transpose - Reshape - Gather - Gather
        do {
            if (node->op_type() == "Reshape" && i + 4 < node_count) {
                if (node_reference[node->output(0)] != 1) {
                    break;
                }

                auto node_transpose           = index_nodes[i + 1].node;
                auto node_reshape2            = index_nodes[i + 2].node;
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i + 2);
                if (next_indexes.size() < 2) {
                    break;
                }
                auto node_gather1 = index_nodes[next_indexes[0]].node;
                auto node_gather2 = index_nodes[next_indexes[1]].node;
                if (node_transpose->op_type() != "Transpose" || node_reshape2->op_type() != "Reshape" ||
                    node_gather1->op_type() != "Gather" || node_gather2->op_type() != "Gather") {
                    break;
                }

                if (node_reference[node->input(0)] != 1 || node_reference[node_transpose->input(0)] != 1 ||
                    node_reference[node_reshape2->input(0)] != 1) {
                    break;
                }

                if (node->output(0) != node_transpose->input(0) ||
                    node_transpose->output(0) != node_reshape2->input(0) ||
                    node_reshape2->output(0) != node_gather1->input(0) ||
                    node_reshape2->output(0) != node_gather2->input(0)) {
                    break;
                }

                std::vector<int64_t> shape1 = get_node_attr_ai(*node, "shape", onnx_net_info_, 1);
                std::vector<int64_t> perm   = get_node_attr_ai(*node_transpose, "perm");
                std::vector<int64_t> shape3 = get_node_attr_ai(*node_reshape2, "shape", onnx_net_info_, 1);

                if (shape1.size() != 3 || perm.size() != 3 || shape3.size() != 5) {
                    break;
                }

                // groups, channels_per_group, height*width
                int64_t output_channels = shape3[2] * 2;
                int64_t group           = shape3[2];

                //                def channel_shuffle_failed(x):
                //                batchsize, num_channels, height, width = x.data.size()
                //                assert (num_channels % 4 == 0)
                //                x = x.reshape(batchsize * num_channels // 2, 2, height * width)
                //                x = x.permute(1, 0, 2)
                //                x = x.reshape(2, -1, num_channels // 2, height, width)
                //                return x[0], x[1]

                //                def channel_shuffle_succeed(x):
                //                batchsize, num_channels, height, width = x.data.size()
                //                assert (num_channels % 4 == 0)
                //                x = x.reshape(batchsize, num_channels // 2, 2, height, width)
                //                x = x.permute(0, 2, 1, 3, 4)
                //                x = x.reshape(batchsize, num_channels, height, width)
                //                return x[:,0:num_channels/2,:,:], x[:,num_channels/2:num_channels,:,:]

                // 1 0 2
                if (perm[0] != 1 || perm[1] != 0 || perm[2] != 2) {
                    break;
                }

                // TODO：考虑情况shape3各种大小
                if (shape3[0] != 2 || shape3[1] != -1) {
                    break;
                }

                int64_t axis1 = get_node_attr_i(*node_gather1, "axis");
                auto indices1 = get_node_attr_ai(*node_gather1, "indices", onnx_net_info_, 1);
                int64_t axis2 = get_node_attr_i(*node_gather2, "axis");
                auto indices2 = get_node_attr_ai(*node_gather2, "indices", onnx_net_info_, 1);
                if (axis1 != 0 || axis2 != 0 || indices1[0] != 0 || indices2[0] != 1) {
                    break;
                }

                // reduce
                node->set_op_type(k_tnn_noop_type);
                node_transpose->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node_transpose->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node_transpose->output(0));

                node_reshape2->set_op_type("ShuffleChannel");
                node_reshape2->set_input(0, node->input(0));

                onnx::AttributeProto* attr_group = node_reshape2->add_attribute();
                attr_group->set_name("group");
                attr_group->set_i(group);

                // convert  gather to stride slice
                node_gather1->set_op_type("Slice");
                {
                    onnx::AttributeProto* attr_starts = node_gather1->add_attribute();
                    attr_starts->set_name("starts");
                    attr_starts->add_ints(0);

                    onnx::AttributeProto* attr_ends = node_gather1->add_attribute();
                    attr_ends->set_name("ends");
                    attr_ends->add_ints(output_channels / 2);

                    onnx::AttributeProto* attr_axes = node_gather1->add_attribute();
                    attr_axes->set_name("axes");
                    attr_axes->add_ints(1);
                }

                node_gather2->set_op_type("Slice");
                {
                    onnx::AttributeProto* attr_starts = node_gather2->add_attribute();
                    attr_starts->set_name("starts");
                    attr_starts->add_ints(output_channels / 2);

                    onnx::AttributeProto* attr_ends = node_gather2->add_attribute();
                    attr_ends->set_name("ends");
                    attr_ends->add_ints(0);

                    onnx::AttributeProto* attr_axes = node_gather2->add_attribute();
                    attr_axes->set_name("axes");
                    attr_axes->add_ints(1);
                }
                i += 4;
            }
        } while (0);

        // ShuffleChannel <= split - unsqueeze(n>=1) - concat - transpose(0,1,2,4,3) - reshape
        do {
            if (node->op_type() == "Split") {
                int64_t group = node->output_size();
                int64_t g     = 0;
                std::vector<onnx::NodeProto*> nodes_unsqueeze;

                for (; g < group; g++) {
                    auto node_unsqueeze = index_nodes[i + 1 + g].node;
                    if (node_unsqueeze->op_type() != "Unsqueeze") {
                        break;
                    }
                    // [BUG] can not get the axis seanxcwang@20200616
                    // else {
                    //     if (get_node_attr_i(*node_unsqueeze, "axis") != 3) {
                    //         std::cout << get_node_attr_i(*node_unsqueeze, "axis") << std::endl;
                    //         break;
                    //     }
                    // }
                    auto next_indexes = GetNextIndexNode(index_nodes, i + 1 + g);
                    if (next_indexes.size() != 1) {
                        break;
                    }
                    nodes_unsqueeze.push_back(node_unsqueeze);
                }
                // all outputs of split should be unsqueeze
                if (g < group)
                    break;

                // next node should be concat
                auto node_concat = index_nodes[i + 1 + group].node;
                if (node_concat->op_type() != "Concat") {
                    break;
                } else {
                    if (get_node_attr_i(*node_concat, "axis") != 3) {
                        break;
                    }
                    auto next_indexes = GetNextIndexNode(index_nodes, i + 1 + group);
                    if (next_indexes.size() != 1) {
                        break;
                    }
                }

                // next node should be transpose
                auto node_transpose = index_nodes[i + 2 + group].node;
                if (node_transpose->op_type() != "Transpose") {
                    break;
                } else {
                    auto next_indexes = GetNextIndexNode(index_nodes, i + 2 + group);
                    if (next_indexes.size() != 1) {
                        break;
                    }
                    std::vector<int64_t> perm = get_node_attr_ai(*node_transpose, "perm");
                    if (perm.size() == 5) {
                        if (perm[0] != 0 || perm[1] != 1 || perm[2] != 2 || perm[3] != 4 || perm[4] != 3) {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                // next node should be reshape
                auto node_reshape = index_nodes[i + 3 + group].node;
                if (node_reshape->op_type() != "Reshape") {
                    break;
                } else {
                    std::vector<int64_t> shape = get_node_attr_ai(*node_reshape, "shape", onnx_net_info_, 1);
                    if (shape.size() != 4) {
                        break;
                    }
                }

                // get the shuffle pattern, reduce now
                node->set_op_type(k_tnn_noop_type);
                for (auto& iter : nodes_unsqueeze) {
                    iter->set_op_type(k_tnn_noop_type);
                }
                node_concat->set_op_type(k_tnn_noop_type);
                node_transpose->set_op_type(k_tnn_noop_type);

                for (g = 0; g < group; g++) {
                    node_reference.erase(node_reference.find(node->output(g)));
                }
                for (auto& iter : nodes_unsqueeze) {
                    node_reference.erase(node_reference.find(iter->output(0)));
                }
                node_reference.erase(node_reference.find(node_concat->output(0)));
                node_reference.erase(node_reference.find(node_transpose->output(0)));

                for (g = 0; g < group; g++) {
                    blob_names.erase(node->output(0));
                }
                for (auto& iter : nodes_unsqueeze) {
                    blob_names.erase(iter->output(0));
                }
                blob_names.erase(node_concat->output(0));
                blob_names.erase(node_transpose->output(0));

                // set new node
                node_reshape->set_op_type("ShuffleChannel");
                node_reshape->set_input(0, node->input(0));
                onnx::AttributeProto* attr_group = node_reshape->add_attribute();
                attr_group->set_name("group");
                attr_group->set_i(group);

                i += 3 + group;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
