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

DECLARE_OP_CONVERTER(DepthToSpace);

string OnnxOpConverterDepthToSpace::TNNOpType(NodeProto &node,
                                                   OnnxNetInfo &net_info) {
    return "Reorg";
}

string OnnxOpConverterDepthToSpace::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    std::string defalt_mode = "CRD";

    int block_size = 2;
    int inverse = 1; // depthtospace convert to reorg inverse mode
    int run_with_output_dims = 0;
    string mode_name = get_node_attr_s(node, "mode", "DCR");
    int mode; // 0 for DCR mode, 1 for CRD mode;
    if (mode_name == "DCR") {
        mode = 0;
    } else {
        mode = 1;
    }

    layer_param << block_size << " " << inverse << " " << run_with_output_dims << " " << mode << " ";
   
    return layer_param.str();
}

int OnnxOpConverterDepthToSpace::WriteTNNModel(serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(DepthToSpace, DepthToSpace);
