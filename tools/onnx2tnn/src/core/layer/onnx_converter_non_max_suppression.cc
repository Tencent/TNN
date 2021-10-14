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

DECLARE_OP_CONVERTER_WITH_FUNC(NonMaxSuppression,
                               virtual std::vector<std::string> GetValidInputNames(NodeProto& node,
                                                                                   OnnxNetInfo& net_info););

string OnnxOpConverterNonMaxSuppression::TNNOpType(NodeProto& node, OnnxNetInfo& net_info) {
    return "NonMaxSuppression";
}

std::vector<std::string> OnnxOpConverterNonMaxSuppression::GetValidInputNames(NodeProto& node, OnnxNetInfo& net_info) {
    return {node.input(0), node.input(1)};
}

string OnnxOpConverterNonMaxSuppression::TNNLayerParam(NodeProto& node, OnnxNetInfo& net_info) {
    int center_point_box = (int)get_node_attr_i(node, "center_point_box", 0);
    int input_size       = node.input_size();

    int64_t max_output_boxes_per_class = 0;
    float iou_threshold                = 0.0f;
    float score_threshold              = 0.0f;

    if (input_size >= 3) {
        auto vec_i = get_node_attr_ai(node, "max_output_boxes_per_class", net_info, 2);
        if (vec_i.size() > 0) {
            max_output_boxes_per_class = vec_i[0];
        }
    }
    if (input_size >= 4) {
        auto vec_f = get_node_attr_af(node, "iou_threshold", net_info, 3);
        if (vec_f.size() > 0) {
            iou_threshold = vec_f[0];
        }
    }
    if (input_size >= 5) {
        auto vec_f = get_node_attr_af(node, "score_threshold", net_info, 4);
        if (vec_f.size() > 0) {
            score_threshold = vec_f[0];
        }
    }

    ostringstream layer_param;
    layer_param << center_point_box << " " << max_output_boxes_per_class << " " << iou_threshold << " "
                << score_threshold << " ";

    return layer_param.str();
}

bool OnnxOpConverterNonMaxSuppression::HasLayerResource(NodeProto& node, OnnxNetInfo& net_info) {
    return false;
}

int OnnxOpConverterNonMaxSuppression::WriteTNNModel(Serializer* net_writer, NodeProto& node, OnnxNetInfo& net_info) {
    // similar to LSTM, from 2020.12.20 on, no WriteTNNModel, no LayerResource, use const_resource_map
    return 0;
}

REGISTER_OP_CONVERTER(NonMaxSuppression, NonMaxSuppression);
