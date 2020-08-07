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

DECLARE_OP_CONVERTER(Resize);

string OnnxOpConverterResize::TNNOpType(NodeProto &node,
                                             OnnxNetInfo &net_info) {
    return "Upsample";
}

string OnnxOpConverterResize::TNNLayerParam(NodeProto &node,
                                                 OnnxNetInfo &net_info) {
    ostringstream layer_param;

    std::string coordinate_transformation_mode =
        get_node_attr_s(node, "coordinate_transformation_mode", "half_pexel");
    std::string mode  = get_node_attr_s(node, "mode");

    std::vector<float> scales;
    std::vector<int64_t> sizes;
    if (net_info.opset >= 11) {
        scales = get_node_attr_af(node, "scales", net_info, 2);
        sizes = get_node_attr_ai(node, "sizes", net_info, 3);
    } else {
        scales = get_node_attr_af(node, "scales", net_info, 1);
    }
    float h_scale = 0;
    float w_scale = 0;

    int resize_type = 0;
    if (mode == "nearest") {
        resize_type = 1;
    } else if (mode == "bilinear" || mode == "linear") {
        resize_type = 2;
    } else if (mode == "trilinear") {
        DLog("not implement\n");
        assert(0);
    }
    
    int align_corners = 0;
    if (coordinate_transformation_mode == "half_pixel") {
        align_corners = 0;
    } else if (coordinate_transformation_mode == "align_corners") {
        align_corners = 1;
    } else {
        DLog("resize: coordinate_transformation_mode(%s) is not supported, result may be different.\n", coordinate_transformation_mode.c_str());
    }
    
    if (sizes.size() <= 0) {
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
            h_scale = get_node_attr_f(node, "height_scale", -1.0f);
            w_scale = get_node_attr_f(node, "width_scale", -1.0f);
            
            if (h_scale <= 0 || w_scale <= 0) {
                DLog("resize invalid scale\n");
                assert(0);
            }
        }

//        if (h_scale <= 1.f || w_scale <= 1.f) {
//            DLog("resize to smaller hw not implemented.\n");
//            assert(0);
//        }
        
        layer_param << resize_type << " " << h_scale << " " << w_scale << " "
                    << align_corners << " ";
    } else {
        h_scale = 0.0;
        w_scale = 0.0;
        
        int target_height = 0;
        int target_width = 0;
        
        if (sizes.size() == 4) {
            target_height =  (int)sizes[2];
            target_width =  (int)sizes[3];
        }
        
        if (target_height <= 0 || target_width <= 0) {
            DLog("resize to smaller hw not implemented.\n");
            assert(0);
        }
        layer_param << resize_type << " " << h_scale << " " << w_scale << " "
                    << align_corners << " "<<target_height<<" "<<target_width<<" ";
    }
    
    return layer_param.str();
}

int OnnxOpConverterResize::WriteTNNModel(serializer *net_writer,
                                              NodeProto &node,
                                              OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Resize, Resize);
