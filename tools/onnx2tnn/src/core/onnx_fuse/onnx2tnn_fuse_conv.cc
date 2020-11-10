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

#include "half_utils.h"
#include "objseri.h"
#include "onnx2tnn.h"

int Onnx2TNN::FuseConv(onnx::GraphProto* mutable_graph,
                       std::vector<IndexNode>& index_nodes,
                       std::map<std::string, onnx::TensorProto>& weights,
                       std::map<std::string, int>& node_reference,
                       std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    std::map<std::string, std::map<std::string, int>> unable_fuse_table;
    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        do {
            if (node->op_type() == "Conv" && i + 1 < node_count) {
                auto node_conv      = node;
                auto node_batchnorm = index_nodes[i + 1].node;

                // check op
                if (!(node_batchnorm->op_type() == "BatchNormalization"))
                    break;
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }

                auto conv_weights_name = node_conv->input(1);
                auto bn_gamma_name = node_batchnorm->input(1);
                if (unable_fuse_table.count(conv_weights_name) == 0) {
                    std::map<std::string, int> tmp;
                    tmp[bn_gamma_name] = 0;
                    unable_fuse_table[conv_weights_name] = std::move(tmp);
                } else if (unable_fuse_table[conv_weights_name].count(bn_gamma_name) == 0) {
                    unable_fuse_table[conv_weights_name][bn_gamma_name] = 0;
                }
                unable_fuse_table[conv_weights_name][bn_gamma_name] ++;

                i += 1;
            }
        } while (0);
    }

    // Conv <= Conv - BatchNormalization
    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;

        do {
            if (node->op_type() == "Conv" && i + 1 < node_count) {
                auto node_conv      = node;
                auto node_batchnorm = index_nodes[i + 1].node;

                // check op
                if (!(node_batchnorm->op_type() == "BatchNormalization"))
                    break;
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }

                auto conv_weights_name = node_conv->input(1);
                auto bn_gamma_name = node_batchnorm->input(1);
                if (unable_fuse_table[conv_weights_name].size() > 1) {
                    break;
                }

                auto kernel_shape =
                    get_node_attr_ai(*node_conv, "kernel_shape");

                bool can_fuse = false;
                if (node_conv->output_size() == 1 &&
                    node_batchnorm->input_size() == 5 &&
                    node_conv->output(0) == node_batchnorm->input(0)) {
                    //目前仅仅考虑二维情况
                    can_fuse = kernel_shape.size() == 2;
                }
                int kernel_size = (int)kernel_shape[0] * kernel_shape[1];

                if (!can_fuse) {
                    break;
                }

                int group = (int)get_node_attr_i(*node_conv, "group", 1);
                auto& conv_weight_tensor = weights[node_conv->input(1)];
                int channel_output       = (int)conv_weight_tensor.dims(0);
                int channel_input = (int)conv_weight_tensor.dims(1) * group;

                float* slope = new float[channel_output];
                float* bias  = new float[channel_output];
                {
                    float epsilon =
                        get_node_attr_f(*node_batchnorm, "epsilon", 1e-5f);

                    const onnx::TensorProto& gamma =
                        weights[node_batchnorm->input(1)];
                    const onnx::TensorProto& beta =
                        weights[node_batchnorm->input(2)];
                    const onnx::TensorProto& mean =
                        weights[node_batchnorm->input(3)];
                    const onnx::TensorProto& var =
                        weights[node_batchnorm->input(4)];

                    int channels = get_tensor_proto_data_size(gamma);
                    assert(channels == channel_output);

                    // apply epsilon to var
                    {
                        const float* gamma_data = get_tensor_proto_data(gamma);
                        const float* beta_data  = get_tensor_proto_data(beta);
                        const float* mean_data  = get_tensor_proto_data(mean);
                        const float* var_data   = get_tensor_proto_data(var);

                        for (int j = 0; j < channels; j++) {
                            double sqrt_var =
                                sqrt(double(var_data[j]) + epsilon);
                            bias[j] = double(beta_data[j]) -
                                      double(gamma_data[j]) *
                                          double(mean_data[j]) / sqrt_var;
                            slope[j] = double(gamma_data[j]) / sqrt_var;
                        }
                    }
                }

                int has_bias = node_conv->input_size() == 3 ? 1 : 0;
                if (!has_bias) {
                    auto temp_tensor =
                        onnx::TensorProto(weights[node_batchnorm->input(2)]);
                    float* temp_tensor_data =
                        get_tensor_proto_mutable_data(temp_tensor);
                    int channels = get_tensor_proto_data_size(temp_tensor);
                    assert(channels == channel_output);
                    for (int j = 0; j < channels; j++) {
                        temp_tensor_data[j] = 0;
                    }
                    auto temp_tensor_name = node_batchnorm->output(0) + "_bias";
                    weights[temp_tensor_name] = temp_tensor;

                    node_conv->add_input(temp_tensor_name);
                }
                auto& conv_bias_tensor = weights[node_conv->input(2)];

                auto new_conv_weight_name = node_conv->input(1) + "_@" + std::to_string(i);
                weights[new_conv_weight_name] = onnx::TensorProto(weights[node_conv->input(1)]);
                auto& new_conv_weight_tensor = weights[new_conv_weight_name];
                node_conv->set_input(1, new_conv_weight_name);

                auto new_conv_bias_name = node_conv->input(2) + "_@" + std::to_string(i);
                weights[new_conv_bias_name] = onnx::TensorProto(weights[node_conv->input(2)]);
                auto& new_conv_bias_tensor = weights[new_conv_bias_name];
                node_conv->set_input(2, new_conv_bias_name);

                // modeify conv weight
                float* conv_weights =
                    get_tensor_proto_mutable_data(new_conv_weight_tensor);
                float* conv_bias =
                    get_tensor_proto_mutable_data(new_conv_bias_tensor);

                const int channel_input_group  = channel_input / group;
                const int channel_output_group = channel_output / group;
                for (int g = 0; g < group; g++) {
                    for (int g_o = 0; g_o < channel_output_group; g_o++) {
                        int oc = g * channel_output_group + g_o;
                        for (int g_i = 0; g_i < channel_input_group; g_i++) {
                            for (int g_k = 0; g_k < kernel_size; g_k++) {
                                int index =
                                    g * channel_output_group *
                                        channel_input_group * kernel_size +
                                    g_o * channel_input_group * kernel_size +
                                    g_i * kernel_size + g_k;
                                conv_weights[index] *= slope[oc];
                            }
                        }
                        conv_bias[oc] = conv_bias[oc] * slope[oc] + bias[oc];
                        //                        conv_bias[oc] = conv_bias[oc]
                        //                        + bias[oc] + 1000;
                    }
                }

                delete[] slope;
                delete[] bias;

                node_batchnorm->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node_conv->output(0)));
                blob_names.erase(node_conv->output(0));
                node_conv->set_output(0, node_batchnorm->output(0));

                i += 1;
            }
        } while (0);

        // Conv <= Conv - Add
        do {
            if (node->op_type() == "Conv" && i + 1 < node_count) {
                auto node_conv = node;
                std::vector<int> next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }
                auto node_add  = index_nodes[next_indexes[0]].node;

                // check op
                if (!(node_add->op_type() == "Add")) {
                    break;
                }
                if (weights.find(node_add->input(0)) == weights.end() &&
                    weights.find(node_add->input(1)) == weights.end()) {
                    // Add don't have weight
                    break;
                }

                auto kernel_shape =
                    get_node_attr_ai(*node_conv, "kernel_shape");
                bool can_fuse = false;
                if (node_conv->output_size() == 1 &&
                    node_add->input_size() == 2 &&
                    node_conv->output(0) == node_add->input(0)) {
                    //目前仅仅考虑二维情况
                    can_fuse = kernel_shape.size() == 2;
                }
                if (!can_fuse) {
                    break;
                }

                int group = (int)get_node_attr_i(*node_conv, "group", 1);
                auto& conv_weight_tensor = weights[node_conv->input(1)];
                int channel_output       = (int)conv_weight_tensor.dims(0);
                // get add weight
                onnx::TensorProto add_bias_tensor;
                std::string add_bias_name;
                if (weights.find(node_add->input(0)) != weights.end()) {
                    add_bias_name   = node_add->input(0);
                    add_bias_tensor = onnx::TensorProto(weights[add_bias_name]);
                } else {
                    add_bias_name   = node_add->input(1);
                    add_bias_tensor = onnx::TensorProto(weights[add_bias_name]);
                }

                int add_bias_size = get_tensor_proto_data_size(add_bias_tensor);
                if (add_bias_size != 1) {
                    if (add_bias_tensor.dims_size() < 2) {
                        break;
                    }
                    int add_bias_channel_size = add_bias_tensor.dims(1);
                    if (add_bias_size != channel_output ||
                        add_bias_channel_size != channel_output) {
                        break;
                    }
                }
                int has_bias = node_conv->input_size() == 3 ? 1 : 0;
                if (!has_bias) {
                    // move add bias to Conv
                    node_conv->add_input(add_bias_name);
                } else {
                    float* add_bias =
                        get_tensor_proto_mutable_data(add_bias_tensor);
                    auto& conv_bias_tensor = weights[node_conv->input(2)];
                    float* conv_bias =
                        get_tensor_proto_mutable_data(conv_bias_tensor);

                    for (int i = 0; i < channel_output; ++i) {
                        if (add_bias_size == 1) {
                            conv_bias[i] = conv_bias[i] + add_bias[0];
                        } else {
                            conv_bias[i] = conv_bias[i] + add_bias[i];
                        }
                    }
                }

                node_add->set_op_type(k_tnn_noop_type);
                node_reference.erase(node_reference.find(node_conv->output(0)));
                blob_names.erase(node_conv->output(0));
                node_conv->set_output(0, node_add->output(0));

                i += 1;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
