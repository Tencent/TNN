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

int Onnx2TNN::FuseBatchNorm(onnx::GraphProto* mutable_graph,
                                 std::vector<IndexNode> & index_nodes,
                                 std::map<std::string, onnx::TensorProto>& weights,
                                 std::map<std::string, int>& node_reference,
                                 std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        // BatchNormalization <= Unsqueeze - BatchNormalization - Squeeze
        do {
            if (node->op_type() == "Unsqueeze")
            {
                if (node_reference.find(node->output(0)) == node_reference.end() || node_reference[node->output(0)] != 1)
                    continue;

                if (i+2 >= node_count)
                    continue;

                auto node2 = index_nodes[i+1].node;
                auto node3 = index_nodes[i+2].node;

                if (node2->op_type() != "BatchNormalization" || node3->op_type() != "Squeeze")
                    continue;

                if (node_reference.find(node2->output(0)) == node_reference.end() || node_reference[node2->output(0)] != 1)
                    continue;

                if (node2->input(0) != node->output(0) || node3->input(0) != node2->output(0))
                    continue;

                // reduce
                node->set_op_type(k_tnn_noop_type);
                node3->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node2->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node2->output(0));

                node2->set_input(0, node->input(0));
                node2->set_output(0, node3->output(0));

                i += 2;
            }
        } while (0);
        
        // BatchNormalization <= Mul - Add
        do {
            if (node->op_type() == "Mul" &&  i + 1 < node_count)
            {
                auto node_mul      = node;
                auto node_add = index_nodes[i + 1].node;

                // check op
                if (!(node_add->op_type() == "Add"))
                    break;
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }

                //check weights
                string scale_name, bias_name, input_name;
                std::vector<int> scale_dims;
                bool is_valid_dims = true;
                int data_count = 1;
                {
                    if (weights.find(node_mul->input(0)) != weights.end() && weights.find(node_mul->input(1)) == weights.end()) {
                        scale_name = node_mul->input(0);
                        input_name = node_mul->input(1);
                    } else if (weights.find(node_mul->input(1)) != weights.end() && weights.find(node_mul->input(0)) == weights.end()) {
                        scale_name = node_mul->input(1);
                        input_name = node_mul->input(0);
                    } else {
                        is_valid_dims = false;
                        break;
                    }
                    
                    if (weights.find(node_add->input(0)) != weights.end() && weights.find(node_add->input(1)) == weights.end()) {
                        bias_name = node_add->input(0);
                    } else if (weights.find(node_add->input(1)) != weights.end() && weights.find(node_add->input(0)) == weights.end()) {
                        bias_name = node_add->input(1);
                    } else {
                        is_valid_dims = false;
                        break;
                    }
                    
                    auto weight_scale = weights[scale_name];
                    auto dims_scale =  weight_scale.dims();
                    
                    auto weight_bias = weights[scale_name];
                    auto dims_bias =  weight_scale.dims();
                    if (weight_bias.dims_size() != dims_bias.size() || dims_bias.size() < 2 || dims_bias.Get(0) != 1) {
                        is_valid_dims = false;
                        break;
                    }
                    
                    for (int ind = 0; ind < dims_scale.size(); ind++) {
                        if (dims_scale.Get(ind) != dims_bias.Get(ind)) {
                            is_valid_dims = false;
                            break;
                        }
                        
                        //check dims to insure no broadcast
                        if (ind < dims_scale.size()-1 && dims_scale.Get(ind) != 1) {
                            is_valid_dims = false;
                            break;
                        }
                        
                        scale_dims.push_back((int)dims_scale.Get(ind));
                        data_count *= dims_scale.Get(ind);
                    }
                }
                if (!is_valid_dims) {
                    break;
                }
                
                string mean_name = node_add->output(0) + "-mean";
                string var_name = node_add->output(0) + "-var";
                //set weights mean var
                onnx::TensorProto tensor_mean, tensor_var;
                tensor_mean.set_name(mean_name);
                tensor_var.set_name(var_name);
                for (int ind=0; ind<scale_dims.size(); ind++) {
                    tensor_mean.add_dims(scale_dims[ind]);
                    tensor_var.add_dims(scale_dims[ind]);
                }
                tensor_mean.set_data_type(onnx::TensorProto_DataType_FLOAT);
                tensor_var.set_data_type(onnx::TensorProto_DataType_FLOAT);
                for (int ind=0; ind<data_count; ind++) {
                    tensor_mean.add_float_data(0);
                    tensor_var.add_float_data(1);
                }
                weights[mean_name] = tensor_mean;
                weights[var_name] = tensor_var;

                // reduce
                node_mul->set_op_type("BatchNormalization");
                node_add->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node_add->output(0)));
                blob_names.erase(node_mul->output(0));
                
                node_mul->clear_input();
                node_mul->add_input(input_name);
                node_mul->add_input(scale_name);
                node_mul->add_input(bias_name);
                node_mul->add_input(mean_name);
                node_mul->add_input(var_name);

                node_mul->set_output(0, node_add->output(0));

                i += 1;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
