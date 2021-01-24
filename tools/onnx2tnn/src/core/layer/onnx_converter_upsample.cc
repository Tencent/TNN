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

DECLARE_OP_CONVERTER(Upsample);

string OnnxOpConverterUpsample::TNNOpType(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    return "Upsample";
}

string OnnxOpConverterUpsample::TNNLayerParam(NodeProto &node,
                                                   OnnxNetInfo &net_info) {
    ostringstream layer_param;

    int align_corners = (int)get_node_attr_i(node, "align_corners", -1);
    std::string mode  = get_node_attr_s(node, "mode");

    std::vector<float> scales;

    if (node.input_size() == 1) {
        scales = get_node_attr_af(node, "scales");
    } else {
        if (net_info.weights_map.find(node.input(1)) != net_info.weights_map.end()) {
            auto &scales_tp = net_info.weights_map[node.input(1)];
            const float *scales_data = get_tensor_proto_data(scales_tp);

            int float_data_size = scales_tp.float_data_size();
            // float data is None, use raw data instead
            if (float_data_size == 0) {
                float_data_size = (int)scales_tp.dims().Get(0);
            }

            for (int j = 0; j < float_data_size; j++) {
                scales.push_back(scales_data[j]);
            }
        }
    }

    int resize_type = 0;
    if (mode == "nearest") {
        resize_type = 1;
    } else if (mode == "bilinear" || mode == "linear") {
        resize_type = 2;
    } else if (mode == "trilinear") {
        DLog("not implement\n");
        assert(0);
    }

    float h_scale = 1.f;
    float w_scale = 1.f;
    if (scales.size() == 2) {
        w_scale = scales[1];
    } else if (scales.size() == 3) {
        h_scale = scales[1];
        w_scale = scales[2];
    } else if (scales.size() == 4) {
        h_scale = scales[2];
        w_scale = scales[3];

        if (scales[1] != 1.f) {
            DLog("not implement\n");
            assert(0);
        }
    } else {
        h_scale = get_node_attr_f(node, "height_scale", 0.0f);
        w_scale = get_node_attr_f(node, "width_scale", 0.0f);
    }

    if (align_corners < 0) {
        if (h_scale >= 1.0f || w_scale >= 1.0f) {
            align_corners = 0;
        } else {
            align_corners = 1;
        }
    }
    layer_param << resize_type << " " << h_scale << " " << w_scale << " "
                << align_corners << " ";

    return layer_param.str();
}

bool OnnxOpConverterUpsample::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterUpsample::WriteTNNModel(Serializer *net_writer,
                                                NodeProto &node,
                                                OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Upsample, Upsample);
