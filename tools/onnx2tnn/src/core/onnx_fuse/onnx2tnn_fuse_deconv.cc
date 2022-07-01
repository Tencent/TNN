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

int Onnx2TNN::FuseDeconv(onnx::GraphProto* mutable_graph,
                              std::vector<IndexNode> & index_nodes,
                              std::map<std::string, onnx::TensorProto>& weights,
                              std::map<std::string, int>& node_reference,
                              std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();
    //ConvTranspose <= ConvTranspose - Add
    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        do {
            if (node->op_type() == "ConvTranspose" && i + 1 < node_count) {
                auto node_deconv = node;
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1 || next_indexes[0] != i+1) {
                    break;
                }
                auto node_add = index_nodes[next_indexes[0]].node;

                // check op
                if (node_add->op_type() != "Add")
                    break;
                
                //check constant
                std::string add_name = "";
                if (weights.find(node_add->input(0)) != weights.end()) {
                    add_name = node_add->input(0);
                } else if (weights.find(node_add->input(1)) != weights.end()) {
                    add_name = node_add->input(1);
                } else {
                    break;
                }
                const onnx::TensorProto& add_tensor = weights[add_name];
                
                //check channel
                int group = (int)get_node_attr_i(*node_deconv, "group", 1);
                auto& deconv_weight_tensor = weights[node_deconv->input(1)];
                int channel_output  = (int)deconv_weight_tensor.dims(1) * group;
                if (channel_output != get_tensor_proto_data_size(add_tensor)) {
                    break;
                }
                
                if (node_deconv->input_size() == 2) {
                    node_deconv->add_input(add_name);
                } else if (node_deconv->input_size() > 2) {
                    auto add_data = get_tensor_proto_data(add_tensor);
                    auto bias_weights = get_tensor_proto_mutable_data(weights[node_deconv->input(2)]);
                    for (int j = 0; j < channel_output; j++) {
                        bias_weights[j] = bias_weights[j] + add_data[j];
                    }
                } else {
                    break;
                }
                
                node_add->set_op_type(k_tnn_noop_type);
                node_reference.erase(node_reference.find(node_deconv->output(0)));
                blob_names.erase(node_deconv->output(0));
                node_deconv->set_output(0, node_add->output(0));
                
                i += 1;
            }
        } while (0);

    }
    ClearEmptyNode(index_nodes);
    
    
    //ConvTranspose <= ConvTranspose - BatchNormalization
    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        do {
            if (node->op_type() == "ConvTranspose" && i + 1 < node_count) {
                auto node_deconv = node;
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }
                auto node_batchnorm = index_nodes[next_indexes[0]].node;

                // check op
                if (!(node_batchnorm->op_type() == "BatchNormalization"))
                    break;

                auto kernel_shape = get_node_attr_ai(*node_deconv, "kernel_shape");

                bool can_fuse = false;
                if (node_deconv->output_size() == 1 && node_batchnorm->input_size() == 5 &&
                    node_deconv->output(0) == node_batchnorm->input(0)) {
                    //目前仅仅考虑二维情况
                    can_fuse = kernel_shape.size() == 2;
                }
                int kernel_size = (int)kernel_shape[0]*kernel_shape[1];

                if (!can_fuse) {
                    break;
                }

                if (node_deconv->input_size() == 2) {
                    int group = (int)get_node_attr_i(*node_deconv, "group", 1);
                    auto& deconv_weight_tensor = weights[node_deconv->input(1)];

                    int channel_input = (int)deconv_weight_tensor.dims(0);
                    int channel_output  = (int)deconv_weight_tensor.dims(1) * group;

                    float* slope = new float[channel_output];
                    float* bias  = new float[channel_output];
                    {
                        float epsilon = get_node_attr_f(*node_batchnorm, "epsilon", 1e-5f);

                        const onnx::TensorProto& gamma = weights[node_batchnorm->input(1)];
                        const onnx::TensorProto& beta  = weights[node_batchnorm->input(2)];
                        const onnx::TensorProto& mean  = weights[node_batchnorm->input(3)];
                        const onnx::TensorProto& var   = weights[node_batchnorm->input(4)];

                        int channels = get_tensor_proto_data_size(gamma);
                        assert(channels == channel_output);


                        // apply epsilon to var
                        {
                            const float* gamma_data = get_tensor_proto_data(gamma);
                            const float* beta_data  = get_tensor_proto_data(beta);
                            const float* mean_data  = get_tensor_proto_data(mean);
                            const float* var_data   = get_tensor_proto_data(var);

                            for (int j = 0; j < channels; j++) {
                                double sqrt_var = sqrt(double(var_data[j])+ epsilon);
                                bias[j] = double(beta_data[j]) - double(gamma_data[j])*double(mean_data[j])/sqrt_var;
                                slope[j]  = double(gamma_data[j])/sqrt_var;
                            }
                        }
                    }

                    //modeify deconv weight
                    float* deconv_weights = get_tensor_proto_mutable_data(deconv_weight_tensor);
                    //float* deconv_bias = get_tensor_proto_mutable_data(deconv_bias_tensor);
                    const int channel_input_group = channel_input / group;
                    const int channel_output_group = channel_output / group;
                    for (int g=0; g<group; g++) {
                        for (int g_o=0; g_o<channel_output_group; g_o++) {
                            int oc = g*channel_output_group+g_o;
                            for (int g_i=0; g_i<channel_input_group; g_i++) {
                                for (int g_k=0; g_k<kernel_size; g_k++) {
                                    int index = g*channel_input_group*channel_output_group*kernel_size +
                                        g_i*channel_output_group*kernel_size +
                                        g_o*kernel_size +
                                        g_k;
                                    deconv_weights[index] *= slope[oc];
                                }
                            }
                            //deconv_bias[oc] = deconv_bias[oc]*slope[oc] + bias[oc];
//                        deconv_bias[oc] = deconv_bias[oc] + bias[oc] + 1000;
                        }
                    }
                    // create tensor proto
                    onnx::TensorProto tensor_B;
                    tensor_B.set_name(node->output(0) + "_B");
                    tensor_B.add_dims(channel_output);
                    tensor_B.set_data_type(onnx::TensorProto_DataType_FLOAT);
                    for (int c = 0; c < channel_output; c++) {
                        tensor_B.add_float_data(bias[c]);
                    }
                    node_deconv->add_input(tensor_B.name());
                    weights[tensor_B.name()] = tensor_B;


                    delete[] slope;
                    delete[] bias;
                    node_batchnorm->set_op_type(k_tnn_noop_type);
                    node_reference.erase(
                        node_reference.find(node_deconv->output(0)));
                    blob_names.erase(node_deconv->output(0));
                    node_deconv->set_output(0, node_batchnorm->output(0));

                    i += 1;
                } else if (node_deconv->input_size() == 3){

                    int group = (int)get_node_attr_i(*node_deconv, "group", 1);
                    auto& deconv_weight_tensor = weights[node_deconv->input(1)];
                    auto& deconv_bias_tensor = weights[node_deconv->input(2)];

                    int channel_input = (int)deconv_weight_tensor.dims(0);
                    int channel_output  = (int)deconv_weight_tensor.dims(1) * group;

                    float* slope = new float[channel_output];
                    float* bias  = new float[channel_output];
                    {
                        float epsilon = get_node_attr_f(*node_batchnorm, "epsilon", 1e-5f);

                        const onnx::TensorProto& gamma = weights[node_batchnorm->input(1)];
                        const onnx::TensorProto& beta  = weights[node_batchnorm->input(2)];
                        const onnx::TensorProto& mean  = weights[node_batchnorm->input(3)];
                        const onnx::TensorProto& var   = weights[node_batchnorm->input(4)];

                        int channels = get_tensor_proto_data_size(gamma);
                        assert(channels == channel_output);


                        // apply epsilon to var
                        {
                            const float* gamma_data = get_tensor_proto_data(gamma);
                            const float* beta_data  = get_tensor_proto_data(beta);
                            const float* mean_data  = get_tensor_proto_data(mean);
                            const float* var_data   = get_tensor_proto_data(var);

                            for (int j = 0; j < channels; j++) {
                                double sqrt_var = sqrt(double(var_data[j])+ epsilon);
                                bias[j] = double(beta_data[j]) - double(gamma_data[j])*double(mean_data[j])/sqrt_var;
                                slope[j]  = double(gamma_data[j])/sqrt_var;
                            }
                        }
                    }

                    //modeify deconv weight
                    float* deconv_weights = get_tensor_proto_mutable_data(deconv_weight_tensor);
                    float* deconv_bias = get_tensor_proto_mutable_data(deconv_bias_tensor);
                    const int channel_input_group = channel_input / group;
                    const int channel_output_group = channel_output / group;
                    for (int g=0; g<group; g++) {
                        for (int g_o=0; g_o<channel_output_group; g_o++) {
                            int oc = g*channel_output_group+g_o;
                            for (int g_i=0; g_i<channel_input_group; g_i++) {
                                for (int g_k=0; g_k<kernel_size; g_k++) {
                                    int index = g*channel_input_group*channel_output_group*kernel_size +
                                        g_i*channel_output_group*kernel_size +
                                        g_o*kernel_size +
                                        g_k;
                                    deconv_weights[index] *= slope[oc];
                                }
                            }
                            deconv_bias[oc] = deconv_bias[oc]*slope[oc] + bias[oc];
//                        deconv_bias[oc] = deconv_bias[oc] + bias[oc] + 1000;
                        }
                    }

                    delete [] slope;
                    delete [] bias;

                    node_batchnorm->set_op_type(k_tnn_noop_type);
                    node_reference.erase(node_reference.find(node_deconv->output(0)));
                    blob_names.erase(node_deconv->output(0));
                    node_deconv->set_output(0, node_batchnorm->output(0));

                    i += 1;
                } else {
                    DLog("error::ConvTranspose node->input_size() == 2 or node->input_size() == 3 ");
                    assert(0);
                }
            }
        } while (0);

    }

    ClearEmptyNode(index_nodes);
    return 0;
}
