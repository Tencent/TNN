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


DECLARE_OP_CONVERTER(Pool);

string OnnxOpConverterPool::TNNOpType(NodeProto &node,
                                           OnnxNetInfo &net_info) {
    std::vector<int64_t> kernel_shape = get_node_attr_ai(node, "kernel_shape");
    const int kernel_shape_size       = kernel_shape.size();
    switch (kernel_shape_size) {
        case 1:
            return "Pooling1D";
        case 2:
            return "Pooling";
        case 3:
            return "Pooling3D";
        default:
            return "Pooling";
    }
}

// NOTE: 由于 Caffe 的 Average Pool 的计算很特殊，与 Pytorch 的 Average Pool 计算不同，
// Caffe 计算 Average Pool 时，除数是 kernel_h * kernel_w
// Pytorch(ONNN) 计算 Average Pool 时，除数是有效位数
// 以上两种计算方式在计算 Average Pool 的边缘时，会造成结果差异。
// 在优化 Average Pool 时，需要注意不能将 caffe 的 Pad + Average Pool 优化掉。
string OnnxOpConverterPool::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    ostringstream layer_param;

    const std::string &onnx_op = node.op_type();
    if (onnx_op == "AveragePool" || onnx_op == "MaxPool") {
        std::string auto_pad = get_node_attr_s(node, "auto_pad");
        std::vector<int64_t> kernel_shape =
            get_node_attr_ai(node, "kernel_shape");
        std::vector<int64_t> strides = get_node_attr_ai(node, "strides");
        std::vector<int64_t> pads    = get_node_attr_ai(node, "pads");
        //计算输出时候采用的截断方式 0：floor 1：ceil
        int ceil_mode = (int)get_node_attr_i(node, "ceil_mode", 0);
        
        int pad_type = -1;
        if (auto_pad == "SAME_UPPER") {
            pad_type = 0;
        } else if (auto_pad == "VALID") {
            pad_type = 1;
        } else if (auto_pad == "SAME_LOWER") {
            pad_type = 0;
            //can be conbine with ceil mode
            DLog("SAME_LOWER is unsuported, change toSAME_UPPER \n");
            assert(0);
        }

        int pool_type = (onnx_op == "AveragePool") ? 1 : 0;
        layer_param << pool_type << " ";

        bool is3d = false;
        if (kernel_shape.size() == 1) {
            layer_param << kernel_shape[0] << " ";
        } else if (kernel_shape.size() == 2) {
            layer_param << kernel_shape[0] << " " << kernel_shape[1] << " ";
        } else if (kernel_shape.size() == 3) {
            is3d = true;
            layer_param << kernel_shape[0] << " " << kernel_shape[1] << " "
                        << kernel_shape[2] << " ";
        }

        if (strides.size() == 1) {
            layer_param << strides[0] << " ";
        } else if (strides.size() == 2) {
            layer_param << strides[0] << " " << strides[1] << " ";
        } else if (strides.size() == 3) {
            layer_param << strides[0] << " " << strides[1] << " " << strides[2]
                        << " ";
        }

        if (pads.size() == 1) {
            layer_param << pads[0] << " ";
        } else if (pads.size() == 2) {
            layer_param << pads[0] << " ";
            pad_type = pads[0] < pads[1] ? 0 : pad_type;
            if (pads[0] > pads[1]) {
                DLog("SAME_LOWER is unsuported, change toSAME_UPPER \n");
                assert(0);
            }
        } else if (pads.size() == 4) {
            if (pads[0] == pads[2] && pads[1] == pads[3]) {
                layer_param << pads[0] << " " << pads[1] << " ";
            } else if (pads[0] < pads[2] && pads[1] < pads[3]) {
                pad_type = 0;//SAME UPPER
                layer_param << pads[0] << " " << pads[1] << " ";
            } else {
                DLog("SAME_LOWER is unsuported, change toSAME_UPPER \n");
                assert(0);
            }
        } else if (pads.size() == 6) {
            if (pads[0] == pads[3] && pads[1] == pads[4] && pads[2] == pads[5]) {
                layer_param << pads[0] << " " << pads[1] << " " << pads[2] << " ";
            } else if (pads[0] < pads[3] && pads[1] < pads[4] && pads[2] < pads[5]) {
                pad_type = 0;//SAME UPPER
                layer_param << pads[0] << " " << pads[1] << " " << pads[2] << " ";
            } else {
                 DLog("SAME_LOWER is unsuported, change toSAME_UPPER \n");
                 assert(0);
            }
        } else {
            if (auto_pad == "" || auto_pad == "SAME_LOWER" || auto_pad == "SAME_UPPER" || auto_pad == "VALID") {
                if (kernel_shape.size() == 3) {
                    layer_param << 0 << " " << 0 << " " << 0 << " ";
                } else {
                    layer_param << 0 << " " << 0 << " ";
                }
            } else {
                DLog("not implement\n");
                assert(0);
            }
        }

        // kernel_h_index_in_input_node_size kernel_w_index_in_input_node_size
        // for runtime kernel size of global pool
        if (is3d) {
            layer_param << -1 << " " << -1 << " " << -1 << " ";
        } else {
            layer_param << -1 << " " << -1 << " ";
        }
        
        //pad type
        layer_param << pad_type << " ";

        //ceil mode, 计算输出时候采用的截断方式 0：floor 1：ceil
        layer_param << ceil_mode << " ";
    } else if (onnx_op == "GlobalAveragePool" || onnx_op == "GlobalMaxPool") {
        std::string auto_pad = get_node_attr_s(node, "auto_pad");  // TODO
        std::vector<int64_t> kernel_shape =
            get_node_attr_ai(node, "kernel_shape");
        std::vector<int64_t> strides = get_node_attr_ai(node, "strides");
        std::vector<int64_t> pads    = get_node_attr_ai(node, "pads");
        //计算输出时候采用的截断方式 0：floor 1：ceil
        int ceil_mode = 0;

        int pool_type = (onnx_op == "GlobalAveragePool") ? 1 : 0;

        if (kernel_shape.size() >= 3) {
            layer_param << pool_type << " 0 0 0 1 1 1 0 0 0 -1 -1 -1 ";
        } else {
            layer_param << pool_type << " 0 0 1 1 0 0 -1 -1 ";
        }

        if (auto_pad == "SAME_LOWER" || auto_pad == "SAME_UPPER") {
            DLog("not implement\n");
            assert(0);
        } else {
            layer_param << -1 << " ";
        }

        layer_param << ceil_mode << " ";

        const int is_adaptive_pool = 0;
        const int output_h         = -1;
        const int output_w         = -1;
        layer_param << is_adaptive_pool << " " << output_h << " " << output_w << " ";
    }

    return layer_param.str();
}

bool OnnxOpConverterPool::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterPool::WriteTNNModel(Serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Pool, MaxPool);
REGISTER_OP_CONVERTER(Pool, AveragePool);
REGISTER_OP_CONVERTER(Pool, GlobalMaxPool);
REGISTER_OP_CONVERTER(Pool, GlobalAveragePool);
