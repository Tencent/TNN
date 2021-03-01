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

#include "onnx_op_converter.h"
#include "onnx_utility.h"

DECLARE_OP_CONVERTER(RoiAlign);

string OnnxOpConverterRoiAlign::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "RoiAlign";
}

string OnnxOpConverterRoiAlign::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    ostringstream layer_param;

    auto mode_str       = get_node_attr_s(node, "mode");
    int mode            = mode_str != "max" ? 1 : 0;
    int output_height   = get_node_attr_i(node, "output_height", 1);
    int output_width    = get_node_attr_i(node, "output_width", 1);
    int sampling_ratio  = get_node_attr_i(node, "sampling_ratio", 0);
    float spatial_scale = get_node_attr_f(node, "spatial_scale", 1.0);

    layer_param << mode << " ";
    layer_param << output_height << " ";
    layer_param << output_width << " ";
    layer_param << sampling_ratio << " ";
    layer_param << spatial_scale << " ";

    return layer_param.str();
}

bool OnnxOpConverterRoiAlign::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterRoiAlign::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(RoiAlign, RoiAlign);
