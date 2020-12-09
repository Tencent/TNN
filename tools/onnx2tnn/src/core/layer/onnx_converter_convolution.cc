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

#include "half_utils.h"

DECLARE_OP_CONVERTER(Conv);

string OnnxOpConverterConv::TNNOpType(NodeProto& node,
                                           OnnxNetInfo &net_info) {
    const std::string& onnx_op = node.op_type();

    std::vector<int64_t> kernel_shape = get_node_attr_ai(node, "kernel_shape");

    if (onnx_op == "Conv") {
        return kernel_shape.size() == 3 ? "Convolution3D" : "Convolution";
    } else if (onnx_op == "ConvTranspose") {
        return kernel_shape.size() == 3 ? "Deconvolution3D" : "Deconvolution";
    }

    return "";
}

string OnnxOpConverterConv::TNNLayerParam(NodeProto& node,
                                               OnnxNetInfo& net_info) {
    ostringstream layer_param;

    const std::string& onnx_op = node.op_type();

    const onnx::TensorProto& weight = net_info.weights_map[node.input(1)];
    int channel_output              = 0;
    int channel_input               = 0;

    int group = (int)get_node_attr_i(node, "group", 1);

    if (onnx_op == "Conv") {
        channel_output = (int)weight.dims(0);
        channel_input  = (int)weight.dims(1) * group;
    } else if (onnx_op == "ConvTranspose") {
        channel_input = (int)weight.dims(0);
        channel_output  = (int)weight.dims(1) * group;
    }
    int has_bias = node.input_size() == 3 ? 1 : 0;
    //        has_bias = 0;

    std::string auto_pad = get_node_attr_s(node, "auto_pad");  // TODO
    std::vector<int64_t> kernel_shape = get_node_attr_ai(node, "kernel_shape");
    std::vector<int64_t> dilations    = get_node_attr_ai(node, "dilations");
    std::vector<int64_t> strides      = get_node_attr_ai(node, "strides");
    std::vector<int64_t> pads         = get_node_attr_ai(node, "pads");
    std::vector<int64_t> output_pads = get_node_attr_ai(node, "output_padding");

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
    
    if (output_pads.size()>0 && output_pads[0] != 0) {
        // output padding conver to pad_type 3 for deconvolution
        pad_type = 3;
    }

    layer_param << group << " " << channel_input << " " << channel_output
                << " ";
    
    //kernel size
    if (kernel_shape.size() == 1) {
        layer_param << kernel_shape[0] << " " << kernel_shape[0] << " ";
    } else if (kernel_shape.size() == 2) {
        layer_param << kernel_shape[0] << " " << kernel_shape[1] << " ";
    } else if (kernel_shape.size() == 3) {
        layer_param << kernel_shape[0] << " " << kernel_shape[1] << " "
                    << kernel_shape[2] << " ";
    }
    
    //stride
    if (strides.size() == 1) {
        layer_param << strides[0] << " " << strides[0] << " ";
    } else if (strides.size() == 2) {
        layer_param << strides[0] << " " << strides[1] << " ";
    } else if (strides.size() == 3) {
        layer_param << strides[0] << " " << strides[1] << " " << strides[2]
                    << " ";
    }
    
    //pad
    if (pads.size() == 1) {
        layer_param << pads[0] << " " << pads[0] << " ";
    } else if (pads.size() == 2) {
        layer_param << pads[0] << " " << pads[1] << " ";
    } else if (pads.size() == 4) {
        if (pads[0] == pads[2] && pads[1] == pads[3]) {
            layer_param << pads[0] << " " << pads[1] << " ";
        } else if (pads[0] < pads[2] || pads[1] < pads[3]) {
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
        if (auto_pad == "SAME_LOWER" || auto_pad == "SAME_UPPER" ||
            auto_pad == "VALID" || auto_pad == "") {
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

    layer_param << has_bias << " " << pad_type << " ";

    if (dilations.size() == 1) {
        layer_param << dilations[0] << " " << dilations[0] << " ";
    } else if (dilations.size() == 2) {
        layer_param << dilations[0] << " " << dilations[1] << " ";
    } else if (dilations.size() == 3) {
        layer_param << dilations[0] << " " << dilations[1] << " "
                    << dilations[2] << " ";
    }

    return layer_param.str();
}

int OnnxOpConverterConv::WriteTNNModel(serializer* net_writer,
                                            NodeProto& node,
                                            OnnxNetInfo& net_info) {
    const std::string& onnx_op = node.op_type();
    std::string name = !node.name().empty() ? node.name() : node.output(0);
    const std::string& tnn_layer_type = TNNOpType(node, net_info);

    //写头信息
    net_writer->put_int(0);  //触发type from string
    net_writer->put_string(tnn_layer_type);
    net_writer->put_string(name);

    //写数据
    //对应conv_layer_datad的反序列化
    net_writer->put_string(name);

    int has_bias = node.input_size() == 3 ? 1 : 0;
    //                has_bias = 0;
    net_writer->put_int(has_bias);

    const onnx::TensorProto& weights = net_info.weights_map[node.input(1)];
    WriteTensorData(weights, net_writer, net_info.data_type);

    if (has_bias) {
        const onnx::TensorProto& bias = net_info.weights_map[node.input(2)];
        WriteTensorData(bias, net_writer, net_info.data_type);
    }

    //有权值写入的返回1， 没有的返回0
    return 1;
}

REGISTER_OP_CONVERTER(Conv, Conv);
REGISTER_OP_CONVERTER(Conv, ConvTranspose);
