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

#include <fstream>
#include <iostream>
#include <sstream>


#include "onnx_op_converter.h"
#include "onnx_utility.h"

DECLARE_OP_CONVERTER(DetectionOutput);

string OnnxOpConverterDetectionOutput::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "DetectionOutput";
}

string OnnxOpConverterDetectionOutput::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    int64_t num_classes                = get_node_attr_i(node, "num_classes");
    int64_t share_location             = get_node_attr_i(node, "share_location");
    int64_t background_label_id        = get_node_attr_i(node, "background_label_id");
    int64_t variance_encoded_in_target = get_node_attr_i(node, "variance_encoded_in_target");
    int64_t code_type                  = get_node_attr_i(node, "code_type");
    int64_t keep_top_k                 = get_node_attr_i(node, "keep_top_k");
    float confidence_threshold         = get_node_attr_f(node, "confidence_threshold");
    float nms_threshold                = get_node_attr_f(node, "nms_threshold");
    int64_t top_k                      = get_node_attr_i(node, "top_k");
    float eta                          = get_node_attr_f(node, "eta");

    layer_param << num_classes << " ";
    layer_param << share_location << " ";
    layer_param << background_label_id << " ";
    layer_param << variance_encoded_in_target << " ";
    layer_param << code_type << " ";
    layer_param << keep_top_k << " ";
    layer_param << confidence_threshold << " ";
    layer_param << nms_threshold << " ";
    layer_param << top_k << " ";
    layer_param << eta << " ";

    return layer_param.str();
}

bool OnnxOpConverterDetectionOutput::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterDetectionOutput::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(DetectionOutput, DetectionOutput);
