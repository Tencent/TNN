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

DECLARE_OP_CONVERTER(PriorBox);

string OnnxOpConverterPriorBox::TNNOpType(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    return "PriorBox";
}

string OnnxOpConverterPriorBox::TNNLayerParam(NodeProto &node,
                                                   OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    std::vector<float> min_sizes = get_node_attr_af(node, "min_sizes");
    std::vector<float> max_sizes = get_node_attr_af(node, "max_sizes");
    int32_t clip                   = get_node_attr_i(node, "clip");
    int32_t flip                   = get_node_attr_i(node, "flip");
    std::vector<float> variances = get_node_attr_af(node, "variances");
    std::vector<float> aspect_ratios =
        get_node_attr_af(node, "aspect_ratios");

    std::vector<int64_t> img_sizes = get_node_attr_ai(node, "img_sizes");
    std::vector<float> steps     = get_node_attr_af(node, "steps");
    float offset                 = get_node_attr_f(node, "offset");

    layer_param << min_sizes.size() << " ";
    for (float min_size : min_sizes) {
        layer_param << min_size << " ";
    }

    layer_param << max_sizes.size() << " ";
    for (float max_size : max_sizes) {
        layer_param << max_size << " ";
    }

    layer_param << clip << " ";
    layer_param << flip << " ";

    layer_param << variances.size() << " ";
    for (float variance : variances) {
        layer_param << variance << " ";
    }

    layer_param << aspect_ratios.size() << " ";
    for (float aspect_ratio : aspect_ratios) {
        layer_param << aspect_ratio << " ";
    }
    // [img_w, img_h]
    assert(img_sizes.size() == 2);
    layer_param << img_sizes[0] << " ";
    layer_param << img_sizes[1] << " ";
    // [step_w, step_h]
    assert(steps.size() == 2);
    layer_param << steps[0] << " ";
    layer_param << steps[1] << " ";

    layer_param << offset << " ";

    return layer_param.str();
}

bool OnnxOpConverterPriorBox::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
};

int OnnxOpConverterPriorBox::WriteTNNModel(Serializer *net_writer,
                                                NodeProto &node,
                                                OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(PriorBox, PriorBox);
