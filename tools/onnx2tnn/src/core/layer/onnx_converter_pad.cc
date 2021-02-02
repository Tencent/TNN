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

DECLARE_OP_CONVERTER(Pad);

string OnnxOpConverterPad::TNNOpType(NodeProto &node, OnnxNetInfo &net_info) {
    std::vector<int64_t> pads = get_node_attr_ai(node, "pads", net_info, 1);
    if (pads.size() == 8) {
        return "Pad";
    } else {
        return "PadV2";
    }
}

string OnnxOpConverterPad::TNNLayerParam(NodeProto &node, OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    std::string mode          = get_node_attr_s(node, "mode");
    std::vector<int64_t> pads = get_node_attr_ai(node, "pads", net_info, 1);
    float const_value         = get_node_attr_f(node, "value", net_info, 2, 0.f);

    int type = 0;
    if (mode == "constant" || mode == "") {
        type = 0;
    } else if (mode == "reflect") {
        DLog("Warning: Pad mode 1(reflect) may not support, depend on device\n");
        // TNN reflect pad has a type of 1
        type = 1;
    } else if (mode == "edge") {
        DLog("Warning: Pad mode 2(edge) may not support, depend on device\n");
        // TNN reflect pad has a type of 1
        type = 2;
    } else {
        DLog("Warning: Pad mode 3 may not support, depend on device\n");
        type = 3;
    }
    
    auto op_type = TNNOpType(node, net_info);
    
    if (op_type == "Pad") {
        if (pads.size() == 10) {
            int64_t pad_t   = pads[2];
            int64_t pad_b   = pads[7];
            int64_t pad_l   = pads[3];
            int64_t pad_r   = pads[8];
            int64_t pad_d_f = pads[4];
            int64_t pad_d_b = pads[9];

            layer_param << "0 0 " << pad_t << " " << pad_b << " " << pad_l << " " << pad_r << " " << pad_d_f << " "
                        << pad_d_b << " 0 0 " << type << " ";
        } else if (pads.size() == 8) {
            int64_t pad_c_b = pads[1];
            int64_t pad_c_e = pads[5];
            int64_t pad_t   = pads[2];
            int64_t pad_b   = pads[6];
            int64_t pad_l   = pads[3];
            int64_t pad_r   = pads[7];
            if ((type == 1 || type == 2) && (pad_c_b != 0 || pad_c_e != 0)) {
                DLog("Pad (edge, reflect) do not support pad in channel!");
                assert(0);
            }
            layer_param << "0 0 " << pad_t << " " << pad_b << " " << pad_l << " " << pad_r << " " << pad_c_b << " "
                        << pad_c_e << " " << type << " ";
        } else if (pads.size() == 6) {
            int64_t pad_c_b = pads[1];
            int64_t pad_c_e = pads[4];
            int64_t pad_t   = pads[2];
            int64_t pad_b   = pads[5];
            int64_t pad_l   = 0;
            int64_t pad_r   = 0;
            layer_param << "0 0 " << pad_t << " " << pad_b << " " << pad_l << " " << pad_r << " " << pad_c_b << " "
                        << pad_c_e << " " << type << " ";
        }
    } else {
        int dim_size = (int)pads.size()/2;
        layer_param << dim_size << " ";
        for (int i=0; i<pads.size(); i++) {
            layer_param << pads[i] << " ";
        }
        layer_param << type << " ";
    }
    layer_param << const_value << " ";

    return layer_param.str();
}

bool OnnxOpConverterPad::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
};

int OnnxOpConverterPad::WriteTNNModel(Serializer *net_writer, NodeProto &node, OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Pad, Pad);
