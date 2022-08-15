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

    //Note: MatMul support matrix B has more than 2 dim size, and matrix B may be not constant, so dont transfer MatMul to Gemm
//    for (int i = 0; i < node_count; i++) {
//        auto node = index_nodes[i].node;
//
//        // Gemm <= MatMul - Add
//         do {
//             if (node->op_type() == "MatMul" && i+1<node_count) {
//                 auto node2 = index_nodes[i+1].node;
//                 if (node2->op_type() != "Add")
//                     break;
//                 std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
//                 if (next_indexes.size() != 1) {
//                     break;
//                 }
//
//                 auto B = get_node_attr_tensor(*node, "B", onnx_net_info_, 1);
//                 auto const h = B.dims(0);
//                 auto const w = B.dims(1);
//                 if (B.dims_size() != 2)
//                     break;
//
//                 auto C = get_node_attr_tensor(*node2, "B", onnx_net_info_, 1);
//                 auto const channel = C.dims(0);
//                 if (C.dims_size() != 1)
//                     break;
//
//                 if (w != channel) {
//                     break;
//                 }
//
//                 // reduce
//                 node->set_op_type(k_tnn_noop_type);
//
//                 node_reference.erase(node_reference.find(node->output(0)));
//                 blob_names.erase(node->output(0));
//
//                 node2->set_op_type("Gemm");
//                 node2->set_input(0, node->input(0));
//
//                 node2->set_input(1, B.name());
//
//                 if (node2->input_size() == 2) {
//                     node2->add_input(C.name());
//                 } else {
//                     node2->set_input(2, C.name());
//                 }
//
//                 onnx::AttributeProto* attr_alpha = node2->add_attribute();
//                 attr_alpha->set_name("alpha");
//                 attr_alpha->set_f(1.f);
//
//                 onnx::AttributeProto* attr_beta = node2->add_attribute();
//                 attr_beta->set_name("beta");
//                 attr_beta->set_f(1.f);
//
//                 onnx::AttributeProto* attr_transA = node2->add_attribute();
//                 attr_transA->set_name("transA");
//                 attr_transA->set_i(0);
//
//                 onnx::AttributeProto* attr_transB = node2->add_attribute();
//                 attr_transB->set_name("transB");
//                 attr_transB->set_i(0);
//
//                 i += 1;
//             }
//         } while (0);
//
//        // Gemm <= MatMul
//         do {
//             if (node->op_type() == "MatMul") {
//                 auto B = get_node_attr_tensor(*node, "B", onnx_net_info_, 1);
//                 if (B.dims_size() == 0) {
//                     break;
//                 }
//                 auto const h = B.dims(0);
//                 auto const channel = B.dims(1);
//                 if (B.dims_size() != 2)
//                     break;
//
//                 node->set_op_type("Gemm");
//                 node->set_input(0, node->input(0));
//                 node->set_input(1, B.name());
//
//                 onnx::AttributeProto* attr_alpha = node->add_attribute();
//                 attr_alpha->set_name("alpha");
//                 attr_alpha->set_f(1.f);
//
//                 onnx::AttributeProto* attr_beta = node->add_attribute();
//                 attr_beta->set_name("beta");
//                 attr_beta->set_f(1.f);
//
//                 onnx::AttributeProto* attr_transA = node->add_attribute();
//                 attr_transA->set_name("transA");
//                 attr_transA->set_i(0);
//
//                 onnx::AttributeProto* attr_transB = node->add_attribute();
//                 attr_transB->set_name("transB");
//                 attr_transB->set_i(0);
//
//                 onnx::AttributeProto* attr_C = node->add_attribute();
//                 attr_C->set_name("C");
//                 {
//                     //https://github.com/pytorch/pytorch/blob/master/caffe2/onnx/helper.h
//                     onnx::TensorProto tensor_C;
//                     tensor_C.set_name("C");
//                     tensor_C.add_dims(channel);
//                     tensor_C.set_data_type(onnx::TensorProto_DataType_FLOAT);
//                     for (int c=0; c<channel; c++) {
//                         tensor_C.add_float_data(0);
//                     }
//
//                     attr_C->mutable_t()->CopyFrom(tensor_C);
//                     attr_C->set_type(AttributeProto::TENSOR);
//                 }
//
//                 i += 0;
//             }
//         } while (0);
//    }
    
    ClearEmptyNode(index_nodes);
    
    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // Gemm <= Gemm - BatchNormalization
         do {
             if (node->op_type() == "Gemm" && i+1<node_count) {
                 std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                 if (next_indexes.size() != 1) {
                     break;
                 }
                 auto node_batchnorm = index_nodes[next_indexes[0]].node;
                 if (node_batchnorm->op_type() != "BatchNormalization")
                     break;
                 
                 int transB    = (int)get_node_attr_i(*node, "transB", 0);
                 auto B = get_node_attr_tensor(*node, "B", onnx_net_info_, 1);
                 auto const h = B.dims(0);
                 auto const w = B.dims(1);
                 if (B.dims_size() != 2)
                     break;

                 auto C = get_node_attr_tensor(*node, "C", onnx_net_info_, 2);
                 auto const channel = C.dims(0);
                 if (C.dims_size() != 1)
                     break;

                 if ((transB && h != channel) || (!transB && w != channel)) {
                     break;
                 }
                 
                 auto B_fused_name   = node->input_size() > 1 ? node->input(1) + "_fused_B" : node->output(0) + "_fused_B";
                 auto C_fused_name   = node->input_size() > 2 ? node->input(2) + "_fused_C" : node->output(0) + "_fused_C";
                 onnx::TensorProto B_fused_tensor(B);
                 onnx::TensorProto C_fused_tensor(C);
                 float* B_fused_data = get_tensor_proto_mutable_data(B_fused_tensor);
                 float* C_fused_data = get_tensor_proto_mutable_data(C_fused_tensor);
                 
                 //convert bn slope and bias with gamma beta mean var
                 float* slope = new float[channel];
                 float* bias  = new float[channel];
                 {
                     float epsilon = get_node_attr_f(*node_batchnorm, "epsilon", 1e-5f);

                     const onnx::TensorProto& gamma = weights[node_batchnorm->input(1)];
                     const onnx::TensorProto& beta  = weights[node_batchnorm->input(2)];
                     const onnx::TensorProto& mean  = weights[node_batchnorm->input(3)];
                     const onnx::TensorProto& var   = weights[node_batchnorm->input(4)];

                     if (channel != get_tensor_proto_data_size(gamma)) {
                         break;
                     }

                     // apply epsilon to var
                     {
                         const float* gamma_data = get_tensor_proto_data(gamma);
                         const float* beta_data  = get_tensor_proto_data(beta);
                         const float* mean_data  = get_tensor_proto_data(mean);
                         const float* var_data   = get_tensor_proto_data(var);

                         for (int j = 0; j < channel; j++) {
                             double sqrt_var = sqrt(double(var_data[j]) + epsilon);
                             bias[j]  = double(beta_data[j]) - double(gamma_data[j]) * double(mean_data[j]) / sqrt_var;
                             slope[j] = double(gamma_data[j]) / sqrt_var;
                         }
                     }
                 }
                 
                 {
                     for (int j = 0; j < channel; j++) {
                         C_fused_data[j] = C_fused_data[j] * slope[j] + bias[j];
                     }
                     
                     if (transB) {
                         for (int i = 0; i < h; i++) {
                             for (int j = 0; j < w; j++) {
                                 B_fused_data[i*w+j] = B_fused_data[i*w+j] * slope[i];
                             }
                         }
                     } else {
                         for (int i = 0; i < h; i++) {
                             for (int j = 0; j < w; j++) {
                                 B_fused_data[i*w+j] = B_fused_data[i*w+j] * slope[j];
                             }
                         }
                     }
                 }
                 delete[] slope;
                 delete[] bias;

                 weights[B_fused_name]   = B_fused_tensor;
                 weights[C_fused_name] = C_fused_tensor;

                 // reduce
                 node->set_op_type(k_tnn_noop_type);

                 node_reference.erase(node_reference.find(node->output(0)));
                 blob_names.erase(node->output(0));

                 node_batchnorm->set_op_type("Gemm");
                 node_batchnorm->clear_input();
                 node_batchnorm->add_input(node->input(0));
                 node_batchnorm->add_input(B_fused_name);
                 node_batchnorm->add_input(C_fused_name);

                 onnx::AttributeProto* attr_alpha = node_batchnorm->add_attribute();
                 attr_alpha->set_name("alpha");
                 attr_alpha->set_f(1.f);

                 onnx::AttributeProto* attr_beta = node_batchnorm->add_attribute();
                 attr_beta->set_name("beta");
                 attr_beta->set_f(1.f);

                 onnx::AttributeProto* attr_transA = node_batchnorm->add_attribute();
                 attr_transA->set_name("transA");
                 attr_transA->set_i(0);

                 onnx::AttributeProto* attr_transB = node_batchnorm->add_attribute();
                 attr_transB->set_name("transB");
                 attr_transB->set_i(transB);

                 i += 1;
             }
         } while (0);
        
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
