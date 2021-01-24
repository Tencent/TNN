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

DECLARE_OP_CONVERTER(Tile);

string OnnxOpConverterTile::TNNOpType(NodeProto &node,
                                           OnnxNetInfo &net_info) {
    return "Repeat";
}

string OnnxOpConverterTile::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    const onnx::TensorProto &repeats = net_info.weights_map[node.input(1)];
    int num_repeats                  = (int)get_tensor_proto_data_size(repeats);
    const int64_t *repeats_data      = repeats.int64_data().data();
    if (num_repeats >= 4) {
        for (int ii = 0; ii < num_repeats; ii++) {
            layer_param << repeats_data[ii] << " ";
        }
    } else {
        DLog(
            "error::Repeat convert failed onnx: unsupported num_repeats = "
            "%d\n",
            num_repeats);
        assert(0);
    }

    return layer_param.str();
}

bool OnnxOpConverterTile::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterTile::WriteTNNModel(Serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Tile, Tile);
