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

#include <cmath>
#include <cstdlib>
#include "onnx_op_converter.h"
#include "onnx_utility.h"

DECLARE_OP_CONVERTER(Clip);

string OnnxOpConverterClip::TNNOpType(NodeProto &node,
                                           OnnxNetInfo &net_info) {
    double min = get_node_attr_f(node, "min", net_info, 1, -DBL_MAX);
    double max = get_node_attr_f(node, "max", net_info, 2, DBL_MAX);
    if (std::fabs(min) <= DBL_EPSILON && std::fabs(max - 6) <= DBL_EPSILON) {
        return "ReLU6";
    } else {
        return "Clip";
    }
}

string OnnxOpConverterClip::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    
    double min = get_node_attr_f(node, "min", net_info, 1, -DBL_MAX);
    double max = get_node_attr_f(node, "max", net_info, 2, DBL_MAX);
    if (std::fabs(min) <= DBL_EPSILON && std::fabs(max - 6) <= DBL_EPSILON) {
        return "";
    } else {
        ostringstream layer_param;
        layer_param << min << " " << max << " ";
        return layer_param.str();
    }
}

bool OnnxOpConverterClip::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterClip::WriteTNNModel(Serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    return 0;
}

REGISTER_OP_CONVERTER(Clip, Clip);
