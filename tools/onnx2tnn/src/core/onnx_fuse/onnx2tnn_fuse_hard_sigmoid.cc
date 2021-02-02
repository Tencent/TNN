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

int Onnx2TNN::FuseHardSigmoid(onnx::GraphProto* mutable_graph,
                                   std::vector<IndexNode> & index_nodes,
                                   std::map<std::string, onnx::TensorProto>& weights,
                                   std::map<std::string, int>& node_reference,
                                   std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // HardSigmoid <= Add(+3) - Clip(0,6) - Div(/6)
        do {
            if (node->op_type() == "Add" && i+2 < node_count) {
                if (node_reference.find(node->output(0)) == node_reference.end() ||
                    node_reference[node->output(0)] != 1)
                     break;

                 if (weights.find(node->input(1)) == weights.end())
                     break;

                 const onnx::TensorProto& add_three = weights[node->input(1)];
                 if (add_three.dims_size() != 0 || get_tensor_proto_data_size(add_three) != 1)
                     break;

                 float constant_add_three = add_three.has_raw_data() ? ((const float*)add_three.raw_data().data())[0] : add_three.float_data().data()[0];
                 if (constant_add_three != 3.f)
                     break;

                 auto node2 = index_nodes[i+1].node;
                 auto node3 = index_nodes[i+2].node;

                 if (node2->op_type() != "Clip" || node3->op_type() != "Div")
                     break;

                 if (node_reference.find(node2->output(0)) == node_reference.end() || node_reference[node2->output(0)] != 1)
                     break;

                float relu6_min = get_node_attr_f(*node2, "min", onnx_net_info_,1, -FLT_MAX);
                float relu6_max = get_node_attr_f(*node2, "max", onnx_net_info_, 2, FLT_MAX);
                 if (relu6_min != 0.f || relu6_max != 6.f)
                     break;

                 if (weights.find(node3->input(1)) == weights.end())
                     break;

                 const onnx::TensorProto& div_six = weights[node3->input(1)];
                 if (div_six.dims_size() != 0 || get_tensor_proto_data_size(div_six) != 1)
                     break;

                 float constant_div_six = div_six.has_raw_data() ? ((const float*)div_six.raw_data().data())[0] : div_six.float_data().data()[0];
                 if (constant_div_six != 6.f)
                     break;
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }
                next_indexes = GetNextIndexNode(index_nodes, i+ 1);
                if (next_indexes.size() != 1) {
                    break;
                }

                 // reduce
                 node->set_op_type(k_tnn_noop_type);
                 node2->set_op_type(k_tnn_noop_type);

                 node_reference.erase(node_reference.find(node->output(0)));
                 node_reference.erase(node_reference.find(node2->output(0)));
                 blob_names.erase(node->output(0));
                 blob_names.erase(node2->output(0));

                 node3->set_op_type("HardSigmoid");
                 node3->clear_input();
                 node3->add_input(node->input(0));

                 onnx::AttributeProto* attr_alpha = node3->add_attribute();
                 attr_alpha->set_name("alpha");
                 attr_alpha->set_f(1.f/6.f);

                 onnx::AttributeProto* attr_beta = node3->add_attribute();
                 attr_beta->set_name("beta");
                 attr_beta->set_f(3.f/6.f);

                 i += 2;
             }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
