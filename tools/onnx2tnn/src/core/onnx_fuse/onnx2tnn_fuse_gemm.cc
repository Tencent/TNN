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

int Onnx2TNN::FuseGEMM(onnx::GraphProto* mutable_graph,
                              std::vector<IndexNode> & index_nodes,
                              std::map<std::string, onnx::TensorProto>& weights,
                              std::map<std::string, int>& node_reference,
                              std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // Gemm <= MatMul - Add
         do {
             if (node->op_type() == "MatMul" && i+1<node_count) {
                 auto node2 = index_nodes[i+1].node;
                 if (node2->op_type() != "Add")
                     break;
                 std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                 if (next_indexes.size() != 1) {
                     break;
                 }

                 auto B = get_node_attr_tensor(*node, "B", onnx_net_info_, 1);
                 auto const h = B.dims(0);
                 auto const w = B.dims(1);
                 if (B.dims_size() != 2)
                     break;

                 auto C = get_node_attr_tensor(*node2, "B", onnx_net_info_, 1);
                 auto const channel = C.dims(0);
                 if (C.dims_size() != 1)
                     break;

                 if (w != channel) {
                     break;
                 }

                 // reduce
                 node->set_op_type(k_tnn_noop_type);

                 node_reference.erase(node_reference.find(node->output(0)));
                 blob_names.erase(node->output(0));

                 node2->set_op_type("Gemm");
                 node2->set_input(0, node->input(0));

                 node2->set_input(1, B.name());

                 if (node2->input_size() == 2) {
                     node2->add_input(C.name());
                 } else {
                     node2->set_input(2, C.name());
                 }

                 onnx::AttributeProto* attr_alpha = node2->add_attribute();
                 attr_alpha->set_name("alpha");
                 attr_alpha->set_f(1.f);

                 onnx::AttributeProto* attr_beta = node2->add_attribute();
                 attr_beta->set_name("beta");
                 attr_beta->set_f(1.f);

                 onnx::AttributeProto* attr_transA = node2->add_attribute();
                 attr_transA->set_name("transA");
                 attr_transA->set_i(0);

                 onnx::AttributeProto* attr_transB = node2->add_attribute();
                 attr_transB->set_name("transB");
                 attr_transB->set_i(0);

                 i += 1;
             }
         } while (0);

        // Gemm <= MatMul
         do {
             if (node->op_type() == "MatMul") {
                 auto B = get_node_attr_tensor(*node, "B", onnx_net_info_, 1);
                 if (B.dims_size() == 0) {
                     break;
                 }
                 auto const h = B.dims(0);
                 auto const channel = B.dims(1);
                 if (B.dims_size() != 2)
                     break;

                 node->set_op_type("Gemm");
                 node->set_input(0, node->input(0));
                 node->set_input(1, B.name());

                 onnx::AttributeProto* attr_alpha = node->add_attribute();
                 attr_alpha->set_name("alpha");
                 attr_alpha->set_f(1.f);

                 onnx::AttributeProto* attr_beta = node->add_attribute();
                 attr_beta->set_name("beta");
                 attr_beta->set_f(1.f);

                 onnx::AttributeProto* attr_transA = node->add_attribute();
                 attr_transA->set_name("transA");
                 attr_transA->set_i(0);

                 onnx::AttributeProto* attr_transB = node->add_attribute();
                 attr_transB->set_name("transB");
                 attr_transB->set_i(0);

                 onnx::AttributeProto* attr_C = node->add_attribute();
                 attr_C->set_name("C");
                 {
                     //https://github.com/pytorch/pytorch/blob/master/caffe2/onnx/helper.h
                     onnx::TensorProto tensor_C;
                     tensor_C.set_name("C");
                     tensor_C.add_dims(channel);
                     tensor_C.set_data_type(onnx::TensorProto_DataType_FLOAT);
                     for (int c=0; c<channel; c++) {
                         tensor_C.add_float_data(0);
                     }

                     attr_C->mutable_t()->CopyFrom(tensor_C);
                     attr_C->set_type(AttributeProto::TENSOR);
                 }

                 i += 0;
             }
         } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
