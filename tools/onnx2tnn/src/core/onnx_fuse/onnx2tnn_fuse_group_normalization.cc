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

#include "onnx2tnn.h"

int Onnx2TNN::FuseGroupNormalization(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                                        std::map<std::string, onnx::TensorProto>& weights,
                                        std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto node_count = index_nodes.size();
    for (int i = 0; i < node_count; i++) {
        auto node_reshape0 = index_nodes[i].node;

        /**
         * Fuse for a special model
         * GroupNormalization <= X + Reshape + InstanceNormalization + Shape + Reshape + (Unsqueeze) + Mul + (Unsqueeze) + Add
         *
         * */
        do {
            string node0_type = node_reshape0->op_type();
            if (node0_type == "Reshape" && i + 5 < node_count) {
                auto node_inst_norm  = index_nodes[i + 1].node;
                auto node_shape  = index_nodes[i + 2].node;
                auto node_reshape  = index_nodes[i + 3].node;
                
                int offset = 5;
                onnx::NodeProto* node_unsqueeze_scale = nullptr;
                onnx::NodeProto* node_mul = nullptr;
                onnx::NodeProto* node_unsqueeze_bias = nullptr;
                onnx::NodeProto* node_add = nullptr;
                if (i + 7 < node_count) {
                    node_unsqueeze_scale = index_nodes[i + 4].node;
                    node_mul = index_nodes[i + 5].node;
                    node_unsqueeze_bias = index_nodes[i + 6].node;
                    node_add = index_nodes[i + 7].node;
                    offset = 7;
                } else {
                    node_mul = index_nodes[i + 4].node;
                    node_add = index_nodes[i + 5].node;
                }
                
                if (node_inst_norm->op_type() != "InstanceNormalization" || node_shape->op_type() != "Shape" || node_reshape->op_type() != "Reshape" ||
                    node_mul->op_type() != "Mul" || node_add->op_type() != "Add" ) {
                    break;
                }
                
                if ((!node_unsqueeze_scale && node_unsqueeze_scale->op_type() != "Unsqueeze") ||
                    (!node_unsqueeze_bias && node_unsqueeze_bias->op_type() != "Unsqueeze" )) {
                    break;
                }
                
                //check Shape input
                if (node_shape->input(0) != node_reshape0->input(0)) {
                    break;
                }

                if ( !(node_inst_norm->input(0) == node_reshape0->output(0) && node_reshape->input(0) == node_inst_norm->output(0) &&
                       node_reshape->input(1) == node_shape->output(0) ) ) {
                    break;
                }
                
                if ( !(node_mul->input(0) == node_reshape->output(0) && node_add->input(0) == node_mul->output(0) ) ) {
                    break;
                }
                
                //inst scale bias
                if ( node_inst_norm->input_size() < 3 || weights.find(node_inst_norm->input(1)) == weights.end() ||
                    weights.find(node_inst_norm->input(2)) == weights.end() ){
                    break;
                }
                //group scale bias
                string group_scale_name;
                if ( node_unsqueeze_scale && weights.find(node_unsqueeze_scale->input(0)) != weights.end()){
                    group_scale_name = node_unsqueeze_scale->input(0);
                } else if ( !node_unsqueeze_scale && weights.find(node_mul->input(1)) != weights.end()) {
                    group_scale_name = node_mul->input(1);
                } else {
                    break;
                }
                string group_bias_name;
                if ( node_unsqueeze_bias && weights.find(node_unsqueeze_bias->input(0)) != weights.end()){
                    group_bias_name = node_unsqueeze_bias->input(0);
                } else if ( !node_unsqueeze_bias && weights.find(node_add->input(1)) != weights.end()) {
                    group_bias_name = node_add->input(1);
                } else {
                    break;
                }
                
                const onnx::TensorProto& inst_scale = weights[node_inst_norm->input(1)];
                const onnx::TensorProto& inst_bias  = weights[node_inst_norm->input(2)];
                const int group = get_tensor_proto_data_size(inst_scale);
                
                onnx::TensorProto& group_scale = weights[group_scale_name];
                onnx::TensorProto& group_bias  = weights[group_bias_name];
                const int channels = get_tensor_proto_data_size(group_scale);
                const int channel_per_group = channels / group;
                
                //fix scale bias
                {
                    const float* inst_scale_data = get_tensor_proto_data(inst_scale);
                    const float* inst_bias_data  = get_tensor_proto_data(inst_bias);
                    float* group_scale_data  = get_tensor_proto_mutable_data(group_scale);
                    float* group_bias_data   = get_tensor_proto_mutable_data(group_bias);

                    for (int j = 0; j < channels; j++) {
                        int inst_index = j / channel_per_group;
                        group_bias_data[j] += group_scale_data[j]*inst_bias_data[inst_index];
                        group_scale_data[j] *= inst_scale_data[inst_index];
                    }
                }
                
                node_inst_norm->set_op_type("GroupNormalization");
                // input
                node_inst_norm->clear_input();
                node_inst_norm->add_input(node_reshape0->input(0));
                node_inst_norm->add_input(group_scale_name);
                node_inst_norm->add_input(group_bias_name);
                
                //output
                node_inst_norm->set_output(0, node_add->output(0));
                
                //group
                auto attr_group = node_inst_norm->add_attribute();
                attr_group->set_name("num_groups");
                attr_group->set_i(group);
                
//                auto attr_channel = node_add->add_attribute();
//                attr_channel->set_name("channel");
//                attr_channel->set_i(channels);
                
                node_reshape0->set_op_type(k_tnn_noop_type);
                node_shape->set_op_type(k_tnn_noop_type);
                node_reshape->set_op_type(k_tnn_noop_type);
                node_mul->set_op_type(k_tnn_noop_type);
                node_add->set_op_type(k_tnn_noop_type);
                if (node_unsqueeze_scale) {
                    node_unsqueeze_scale->set_op_type(k_tnn_noop_type);
                    node_reference.erase(node_reference.find(node_unsqueeze_scale->output(0)));
                    blob_names.erase(node_unsqueeze_scale->output(0));
                }
                if (node_unsqueeze_bias) {
                    node_unsqueeze_bias->set_op_type(k_tnn_noop_type);
                    node_reference.erase(node_reference.find(node_unsqueeze_bias->output(0)));
                    blob_names.erase(node_unsqueeze_bias->output(0));
                }

                node_reference.erase(node_reference.find(node_reshape0->output(0)));
                node_reference.erase(node_reference.find(node_shape->output(0)));
                node_reference.erase(node_reference.find(node_reshape->output(0)));
                node_reference.erase(node_reference.find(node_mul->output(0)));
//                node_reference.erase(node_reference.find(node_add->output(0)));
                blob_names.erase(node_reshape0->output(0));
                blob_names.erase(node_inst_norm->output(0));
                blob_names.erase(node_shape->output(0));
                blob_names.erase(node_reshape->output(0));
                blob_names.erase(node_mul->output(0));
                i += offset;
            }
        } while (0);
    }
    ClearEmptyNode(index_nodes);
    
    
    
    //right now group norm is not implemented on device of arm x86 opencl metal and cuda, so return it
    return 0;
    node_count = index_nodes.size();
    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;
        //GroupNormalization <= Reshape(0, group, -1) + InstanceNormalization + Reshape(0, channel, xx, xx) + Mul + Add
        do {
            string reduce_type = node->op_type();
            if (reduce_type == "Reshape" && i + 4 < node_count) {
                auto node_reshape1 = node;
                auto shape1 = get_node_attr_ai(*node_reshape1, "shape", weights, 1);
                if (shape1.size() <= 2) {
                    break;
                }
                const int group = (int)shape1[1];
                
                auto next_indexes = GetNextIndexNode(index_nodes, i);
                if (next_indexes.size() != 1) {
                    break;
                }
                auto node_instnorm = index_nodes[next_indexes[0]].node;
                if (node_instnorm->op_type() != "InstanceNormalization")
                    break;
                
                next_indexes = GetNextIndexNode(index_nodes, next_indexes[0]);
                if (next_indexes.size() != 1) {
                    break;
                }
                auto node_reshape2 = index_nodes[next_indexes[0]].node;
                if (node_reshape2->op_type() != "Reshape")
                    break;
                auto shape2 = get_node_attr_ai(*node_reshape2, "shape", weights, 1);
                if (shape2.size() <= 2) {
                    break;
                }
                const int channels = (int)shape2[1];
                
                next_indexes = GetNextIndexNode(index_nodes, next_indexes[0]);
                if (next_indexes.size() != 1) {
                    break;
                }
                auto node_mul = index_nodes[next_indexes[0]].node;
                if (node_mul->op_type() != "Mul")
                    break;
                
                next_indexes = GetNextIndexNode(index_nodes, next_indexes[0]);
                if (next_indexes.size() != 1) {
                    break;
                }
                auto node_add = index_nodes[next_indexes[0]].node;
                if (node_add->op_type() != "Add")
                    break;
                
                if (node_reshape1->output(0) != node_instnorm->input(0) ||
                    node_instnorm->output(0) != node_reshape2->input(0) ||
                    node_reshape2->output(0) != node_mul->input(0) ||
                    node_mul->output(0) != node_add->input(0)) {
                    break;
                }
                
                //inst scale bias
                if ( node_instnorm->input_size() < 3 || weights.find(node_instnorm->input(1)) == weights.end() ||
                    weights.find(node_instnorm->input(2)) == weights.end() ){
                    break;
                }
                const onnx::TensorProto& inst_scale = weights[node_instnorm->input(1)];
                const onnx::TensorProto& inst_bias  = weights[node_instnorm->input(2)];
                if( group != get_tensor_proto_data_size(inst_scale) ) {
                    break;
                }
                
                //group scale bias
                string group_scale_name = node_mul->input(1);
                string group_bias_name = node_add->input(1);
                if ( weights.find(group_scale_name) == weights.end() ||
                    weights.find(group_bias_name) == weights.end() ){
                    break;
                }
      
                onnx::TensorProto& group_scale = weights[group_scale_name];
                onnx::TensorProto& group_bias  = weights[group_bias_name];
                if (channels != get_tensor_proto_data_size(group_scale)) {
                    break;
                }
                const int channel_per_group = channels / group;
                //fix scale bias
                {
                    const float* inst_scale_data = get_tensor_proto_data(inst_scale);
                    const float* inst_bias_data  = get_tensor_proto_data(inst_bias);
                    float* group_scale_data  = get_tensor_proto_mutable_data(group_scale);
                    float* group_bias_data   = get_tensor_proto_mutable_data(group_bias);

                    for (int j = 0; j < channels; j++) {
                        int inst_index = j / channel_per_group;
                        group_bias_data[j] += group_scale_data[j]*inst_bias_data[inst_index];
                        group_scale_data[j] *= inst_scale_data[inst_index];
                    }
                }
                
                node_reshape1->set_op_type(k_tnn_noop_type);
                node_instnorm->set_op_type(k_tnn_noop_type);
                node_reshape2->set_op_type(k_tnn_noop_type);
                node_mul->set_op_type(k_tnn_noop_type);

                node_reference.erase(node_reference.find(node_reshape1->output(0)));
                node_reference.erase(node_reference.find(node_instnorm->output(0)));
                node_reference.erase(node_reference.find(node_reshape2->output(0)));
                node_reference.erase(node_reference.find(node_mul->output(0)));
                
                blob_names.erase(node_reshape1->output(0));
                blob_names.erase(node_instnorm->output(0));
                blob_names.erase(node_reshape2->output(0));
                blob_names.erase(node_mul->output(0));
                
                node_add->set_op_type("GroupNormalization");
                // input
                node_add->clear_input();
                node_add->add_input(node_reshape1->input(0));
                node_add->add_input(group_scale_name);
                node_add->add_input(group_bias_name);
                
                auto attr_group = node_add->add_attribute();
                attr_group->set_name("num_groups");
                attr_group->set_i(group);
                auto attr_epsilon = node_add->add_attribute();
                attr_epsilon->set_name("epsilon");
                attr_epsilon->set_f(get_node_attr_f(*node_instnorm, "epsilon"));
//                auto attr_channel = node_add->add_attribute();
//                attr_channel->set_name("channel");
//                attr_channel->set_i(channels);
                
                i += 4;
            }
        } while (0);
    }
    ClearEmptyNode(index_nodes);
    return 0;
}
