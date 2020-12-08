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

#include <algorithm>

#include "objseri.h"
#include "onnx2tnn.h"


inline bool IsEqual(float num1, float num2) {
    return std::abs(num1 - num2) <= 1e-6;
}


int Onnx2TNN::FuseHardSwish(onnx::GraphProto* mutable_graph, std::vector<IndexNode>& index_nodes,
                            std::map<std::string, onnx::TensorProto>& weights,
                            std::map<std::string, int>& node_reference, std::set<std::string>& blob_names) {
    auto const node_count = index_nodes.size();

    for (int i = 0; i < node_count; i++) {
        auto node = index_nodes[i].node;
        // HardSwish <= Add - Clip - Div - Mul
        // out =  x0 * (clip(x0 + 3, 0, 6) / 6)
        // out =  x0 * clip(x0/6 + 3/6, 0, 1)
        // ensure HardSigmoid first called before FuseHardSwish, so this pattern never happen
        do {
            if (node->op_type() == "Add" && i + 3 < node_count) {
                if (node_reference.find(node->output(0)) == node_reference.end() ||
                    node_reference[node->output(0)] != 1)
                    break;

                if (weights.find(node->input(1)) == weights.end())
                    break;

                const onnx::TensorProto& add_three = weights[node->input(1)];
                if (add_three.dims_size() != 0 || get_tensor_proto_data_size(add_three) != 1)
                    break;

                float constant_add_three = add_three.has_raw_data() ? ((const float*)add_three.raw_data().data())[0]
                                                                    : add_three.float_data().data()[0];
                if (!IsEqual(constant_add_three, 3.f))
                    break;

                auto node_clip = index_nodes[i + 1].node;
                auto node_div  = index_nodes[i + 2].node;
                auto node_mul  = index_nodes[i + 3].node;

                if (node_clip->op_type() != "Clip" || node_div->op_type() != "Div" || node_mul->op_type() != "Mul")
                    break;

                if (node_reference.find(node_clip->output(0)) == node_reference.end() ||
                    node_reference[node_clip->output(0)] != 1)
                    break;

                if (node_reference.find(node_mul->output(0)) == node_reference.end() ||
                    node_reference[node_mul->output(0)] != 1)
                    break;

                float relu6_min = get_node_attr_f(*node_clip, "min", onnx_net_info_, 1, -FLT_MAX);
                float relu6_max = get_node_attr_f(*node_clip, "max", onnx_net_info_, 2, FLT_MAX);
                if (!IsEqual(relu6_min, 0.f) || !IsEqual(relu6_max, 6.f))
                    break;

                if (!(node_div->input_size() == 2 && node_div->input(0) == node_clip->output(0)))
                    break;

                if (weights.find(node_div->input(1)) == weights.end())
                    break;

                const onnx::TensorProto& div_six = weights[node_div->input(1)];
                if (div_six.dims_size() != 0 || get_tensor_proto_data_size(div_six) != 1)
                    break;

                float constant_div_six = div_six.has_raw_data() ? ((const float*)div_six.raw_data().data())[0]
                                                                : div_six.float_data().data()[0];
                if (!IsEqual(constant_div_six, 6.f))
                    break;
                int x0_index = (node_mul->input(1) == node_div->output(0)) ? 0 : 1;
                std::vector<std::string> inputs;
                inputs.push_back(node_mul->input(x0_index));
                if (inputs[0] != node->input(0)) {
                    inputs.push_back(node->input(0));
                }
                // reduce
                node->set_op_type(k_tnn_noop_type);
                node_clip->set_op_type(k_tnn_noop_type);
                node_div->set_op_type(k_tnn_noop_type);

                node_reference[node->input(0)] -= 1;

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node_clip->output(0)));
                node_reference.erase(node_reference.find(node_div->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node_clip->output(0));
                blob_names.erase(node_div->output(0));

                node_mul->set_op_type("HardSwish");
                node_mul->clear_input();
                node_mul->add_input(inputs[0]);
                if (inputs.size() == 2) {
                    node_mul->add_input(inputs[1]);
                }

                onnx::AttributeProto* attr_alpha = node_mul->add_attribute();
                attr_alpha->set_name("alpha");
                attr_alpha->set_f(1.f / 6.f);

                onnx::AttributeProto* attr_beta = node_mul->add_attribute();
                attr_beta->set_name("beta");
                attr_beta->set_f(3.f / 6.f);

                i += 3;
            }
        } while (0);

        // HardSwish <= Add - Clip - Cast - Div - Cast - Mul
        // out =  x0 * (clip(x0 + 3, 0, 6) / 6)
        // out =  x0 * clip(x0/6 + 3/6, 0, 1)
        do {
            if (node->op_type() == "Add" && i + 6 < node_count) {
                if (node_reference.find(node->output(0)) == node_reference.end() ||
                    node_reference[node->output(0)] != 1)
                    break;

                if (weights.find(node->input(1)) == weights.end())
                    break;

                const onnx::TensorProto& add_three = weights[node->input(1)];
                if (add_three.dims_size() != 0 || get_tensor_proto_data_size(add_three) != 1)
                    break;

                float constant_add_three = add_three.has_raw_data() ? ((const float*)add_three.raw_data().data())[0]
                                                                    : add_three.float_data().data()[0];
                if (!IsEqual(constant_add_three, 3.f))
                    break;

                auto node_clip   = index_nodes[i + 1].node;
                auto node_cast_1 = index_nodes[i + 2].node;
                auto node_div    = index_nodes[i + 3].node;
                auto node_cast_2 = index_nodes[i + 4].node;
                auto node_mul    = index_nodes[i + 5].node;

                if (node_clip->op_type() != "Clip" || node_cast_1->op_type() != "Cast" ||
                    node_div->op_type() != "Div" || node_cast_2->op_type() != "Cast" || node_mul->op_type() != "Mul")
                    break;

                if (node_reference.find(node_clip->output(0)) == node_reference.end() ||
                    node_reference[node_clip->output(0)] != 1)
                    break;

                if (node_reference.find(node_mul->output(0)) == node_reference.end())
                    break;

                float relu6_min = get_node_attr_f(*node_clip, "min", onnx_net_info_, 1, -FLT_MAX);
                float relu6_max = get_node_attr_f(*node_clip, "max", onnx_net_info_, 2, FLT_MAX);
                if (!IsEqual(relu6_min, 0.f) || !IsEqual(relu6_max, 6.f))
                    break;

                if (!(node_div->input_size() == 2 && node_div->input(0) == node_cast_1->output(0)))
                    break;

                if (weights.find(node_div->input(1)) == weights.end())
                    break;

                const onnx::TensorProto& div_six = weights[node_div->input(1)];
                if (div_six.dims_size() != 0 || get_tensor_proto_data_size(div_six) != 1)
                    break;

                float constant_div_six = 0.f;
                if (div_six.has_raw_data()) {
                    auto data_type = div_six.data_type();
                    if (data_type == onnx::TensorProto_DataType_FLOAT) {
                        constant_div_six = ((const float*)div_six.raw_data().data())[0];
                    } else if (data_type == onnx::TensorProto_DataType_DOUBLE) {
                        constant_div_six = (float)((const double*)div_six.raw_data().data())[0];
                    }
                } else {
                    constant_div_six = div_six.float_data().data()[0];
                }
                if (!IsEqual(constant_div_six, 6.f))
                    break;
                int x0_index = (node_mul->input(1) == node_cast_2->output(0)) ? 0 : 1;
                std::vector<std::string> inputs;
                inputs.push_back(node_mul->input(x0_index));
                if (inputs[0] != node->input(0)) {
                    inputs.push_back(node->input(0));
                }
                // reduce
                node->set_op_type(k_tnn_noop_type);
                node_clip->set_op_type(k_tnn_noop_type);
                node_cast_1->set_op_type(k_tnn_noop_type);
                node_div->set_op_type(k_tnn_noop_type);
                node_cast_2->set_op_type(k_tnn_noop_type);

                node_reference[node->input(0)] -= 1;

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node_clip->output(0)));
                node_reference.erase(node_reference.find(node_cast_1->output(0)));
                node_reference.erase(node_reference.find(node_div->output(0)));
                node_reference.erase(node_reference.find(node_cast_2->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node_clip->output(0));
                blob_names.erase(node_cast_1->output(0));
                blob_names.erase(node_div->output(0));
                blob_names.erase(node_cast_2->output(0));

                node_mul->set_op_type("HardSwish");
                node_mul->clear_input();
                node_mul->add_input(inputs[0]);
                if (inputs.size() == 2) {
                    node_mul->add_input(inputs[1]);
                }

                onnx::AttributeProto* attr_alpha = node_mul->add_attribute();
                attr_alpha->set_name("alpha");
                attr_alpha->set_f(1.f / 6.f);

                onnx::AttributeProto* attr_beta = node_mul->add_attribute();
                attr_beta->set_name("beta");
                attr_beta->set_f(3.f / 6.f);

                i += 5;
            }
        } while (0);

        // HardSwish <= Add - Clip - Mul - Div
        // out =  (x0 * clip(x1 + 3, 0, 6)) / 6
        // out =  x0 * clip(x1/6 + 3/6, 0, 1)
        do {
            if (node->op_type() == "Add" && i + 3 < node_count) {
                if (node_reference.find(node->output(0)) == node_reference.end() ||
                    node_reference[node->output(0)] != 1)
                    break;

                if (weights.find(node->input(1)) == weights.end())
                    break;

                const onnx::TensorProto& add_three = weights[node->input(1)];
                if (add_three.dims_size() != 0 || get_tensor_proto_data_size(add_three) != 1)
                    break;

                float constant_add_three = add_three.has_raw_data() ? ((const float*)add_three.raw_data().data())[0]
                                                                    : add_three.float_data().data()[0];
                if (!IsEqual(constant_add_three, 3.f))
                    break;

                auto node_clip = index_nodes[i + 1].node;
                auto node_mul  = index_nodes[i + 2].node;
                auto node_div  = index_nodes[i + 3].node;

                if (node_clip->op_type() != "Clip" || node_mul->op_type() != "Mul" || node_div->op_type() != "Div")
                    break;

                if (node_reference.find(node_clip->output(0)) == node_reference.end() ||
                    node_reference[node_clip->output(0)] != 1)
                    break;

                if (node_reference.find(node_mul->output(0)) == node_reference.end() ||
                    node_reference[node_mul->output(0)] != 1)
                    break;

                float relu6_min = get_node_attr_f(*node_clip, "min", onnx_net_info_, 1, -FLT_MAX);
                float relu6_max = get_node_attr_f(*node_clip, "max", onnx_net_info_, 2, FLT_MAX);
                if (!IsEqual(relu6_min, 0.f) || !IsEqual(relu6_max, 6.f))
                    break;

                if (!(node_mul->input_size() == 2 &&
                      (node_mul->input(0) == node_clip->output(0) || node_mul->input(1) == node_clip->output(0))))
                    break;
                int x0_index = (node_mul->input(1) == node_clip->output(0)) ? 0 : 1;
                std::vector<std::string> inputs;
                inputs.push_back(node_mul->input(x0_index));
                if (inputs[0] != node->input(0)) {
                    inputs.push_back(node->input(0));
                }

                if (node_div->input(0) != node_mul->output(0))
                    break;

                if (weights.find(node_div->input(1)) == weights.end())
                    break;

                const onnx::TensorProto& div_six = weights[node_div->input(1)];
                if (div_six.dims_size() != 0 || get_tensor_proto_data_size(div_six) != 1)
                    break;

                float constant_div_six = div_six.has_raw_data() ? ((const float*)div_six.raw_data().data())[0]
                                                                : div_six.float_data().data()[0];
                if (!IsEqual(constant_div_six, 6.f))
                    break;

                // reduce
                node->set_op_type(k_tnn_noop_type);
                node_clip->set_op_type(k_tnn_noop_type);
                node_mul->set_op_type(k_tnn_noop_type);

                node_reference[node->input(0)] -= 1;

                node_reference.erase(node_reference.find(node->output(0)));
                node_reference.erase(node_reference.find(node_clip->output(0)));
                node_reference.erase(node_reference.find(node_mul->output(0)));
                blob_names.erase(node->output(0));
                blob_names.erase(node_clip->output(0));
                blob_names.erase(node_mul->output(0));

                node_div->set_op_type("HardSwish");
                node_div->clear_input();
                node_div->add_input(inputs[0]);
                if (inputs.size() == 2) {
                    node_div->add_input(inputs[1]);
                }

                onnx::AttributeProto* attr_alpha = node_div->add_attribute();
                attr_alpha->set_name("alpha");
                attr_alpha->set_f(1.f / 6.f);

                onnx::AttributeProto* attr_beta = node_div->add_attribute();
                attr_beta->set_name("beta");
                attr_beta->set_f(3.f / 6.f);

                i += 3;
            }
        } while (0);
        // HardSwish <= HardSigmoid - Mul
        do {
            if (node->op_type() == "HardSigmoid" && i + 1 < node_count) {
                if (node_reference.find(node->output(0)) == node_reference.end() ||
                    node_reference[node->output(0)] != 1)
                    break;

                float alpha = get_node_attr_f(*node, "alpha", 0.2f);
                float beta  = get_node_attr_f(*node, "beta", 0.5f);

                auto node_mul = index_nodes[i + 1].node;

                if (node_mul->op_type() != "Mul")
                    break;

                if (!(node_mul->input_size() == 2 &&
                      (node_mul->input(0) == node->output(0) || node_mul->input(1) == node->output(0))))
                    break;

                int x0_index = (node_mul->input(1) == node->output(0)) ? 0 : 1;
                std::vector<std::string> inputs;
                inputs.push_back(node_mul->input(x0_index));
                if (inputs[0] != node->input(0)) {
                    inputs.push_back(node->input(0));
                }

                // reduce
                node->set_op_type(k_tnn_noop_type);

                node_reference[node->input(0)] -= 1;

                node_reference.erase(node_reference.find(node->output(0)));
                blob_names.erase(node->output(0));

                node_mul->set_op_type("HardSwish");
                node_mul->clear_input();
                node_mul->add_input(inputs[0]);
                if (inputs.size() == 2) {
                    node_mul->add_input(inputs[1]);
                }

                onnx::AttributeProto* attr_alpha = node_mul->add_attribute();
                attr_alpha->set_name("alpha");
                attr_alpha->set_f(alpha);

                onnx::AttributeProto* attr_beta = node_mul->add_attribute();
                attr_beta->set_name("beta");
                attr_beta->set_f(beta);

                i += 1;
            }
        } while (0);
    }

    ClearEmptyNode(index_nodes);
    return 0;
}
