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

DECLARE_OP_CONVERTER(AdaptivePool);

string OnnxOpConverterAdaptivePool::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    return "Pooling";
}

string OnnxOpConverterAdaptivePool::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    ostringstream layer_param;

    const std::string &onnx_op = node.op_type();
    int pool_type              = (onnx_op == "adaptive_avg_pool2d") ? 1 : 0;
    layer_param << pool_type << " ";

    const int kernel_shape     = -1;
    const int stride           = -1;
    const int pad              = 0;
    const int pad_type         = -1;
    const int ceil_mode        = 0;
    const int is_adaptive_pool = 1;
    layer_param << kernel_shape << " " << kernel_shape << " ";
    layer_param << stride << " " << stride << " ";
    layer_param << pad << " " << pad << " ";
    layer_param << -1 << " " << -1 << " ";
    layer_param << pad_type << " ";
    layer_param << ceil_mode << " ";
    layer_param << is_adaptive_pool << " ";

    const auto output_shape_name = node.input(1);
    const auto output_shape      = net_info.weights_map[output_shape_name];
    auto shape_data              = (const int64_t *)get_tensor_proto_data(output_shape);
    int data_size                = get_tensor_proto_data_size(output_shape);
    if (data_size == 1) {
        layer_param << shape_data[0] << " " << shape_data[0] << " ";
    } else if (data_size == 2) {
        layer_param << shape_data[0] << " " << shape_data[1] << " ";
    } else {
        DLog("output shape's size = %d, it's unsuported \n", data_size);
        assert(0);
    }

    return layer_param.str();
}

bool OnnxOpConverterAdaptivePool::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterAdaptivePool::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    return 0;
}

REGISTER_OP_CONVERTER(AdaptivePool, adaptive_avg_pool2d);
