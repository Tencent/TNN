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

DECLARE_OP_CONVERTER(Slice);

string OnnxOpConverterSlice::TNNOpType(NodeProto &node,
                                            OnnxNetInfo &net_info) {
    // MARK:由于CPU版本slice_layer和GPU版本的slice_layer实现不一致，统一为StridedSlice，只是stride为1
    return "StridedSlice";
}

string OnnxOpConverterSlice::TNNLayerParam(NodeProto &node,
                                                OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    std::vector<int64_t> starts = get_node_attr_ai(node, "starts", net_info, 1);
    std::vector<int64_t> ends   = get_node_attr_ai(node, "ends", net_info, 2);
    std::vector<int64_t> axes   = get_node_attr_ai(node, "axes", net_info, 3);
    // Opset 9 Slice层没有steps 属性，部分模型使用HardTanh，opset11 暂不支持。
    std::vector<int64_t> steps;
    if (net_info.opset >= 11) {
        steps = get_node_attr_ai(node, "steps", net_info, 4);
    } else {
        steps = {1, 1, 1, 1};
    }



//    int steps_count             = (int)steps.size();
//    for (int ii = 0; ii < steps_count; ii++) {
//        if (steps[ii] != 1) {
//            DLog("error::Slice convert failed onnx:%s (unsupported steps)\n",
//                 onnx_op.c_str());
//            assert(0);
//        }
//    }
    
    for (int ii = 0; ii < axes.size(); ii++) {
        if (axes[ii] >= INT_MAX) {
            axes[ii] = 1;
        }
        if (axes[ii] <= INT_MIN) {
            axes[ii] = 1;
        }
    }

    int dimension = 4;
    std::vector<int> all_starts, all_ends, all_steps;
    for (int ii = 0; ii < dimension; ii++) {
        all_starts.push_back(0);
        all_ends.push_back(0);
        all_steps.push_back(1);
    }

    for (int ii = 0; ii < axes.size(); ii++) {
        all_starts[axes[ii]] = (int)starts[ii];
        if (ends[ii] >= INT_MAX) {  // 9223372036854775807ll
            all_ends[axes[ii]] = 0;
        } else {
            all_ends[axes[ii]] = (int)ends[ii];
        }

        if (ii < steps.size()) {
            all_steps[axes[ii]] = (int)steps[ii];
        }
    }

    //输出参数
    layer_param << all_starts.size() << " ";
    for (int ii = 0; ii < all_starts.size(); ii++) {
        layer_param << all_starts[ii] << " ";
    }
    layer_param << all_ends.size() << " ";
    for (int ii = 0; ii < all_ends.size(); ii++) {
        layer_param << all_ends[ii] << " ";
    }
    layer_param << all_steps.size() << " ";
    for (int ii = 0; ii < all_steps.size(); ii++) {
        layer_param << all_steps[ii] << " ";
    }

    return layer_param.str();
}

int OnnxOpConverterSlice::WriteTNNModel(serializer *net_writer,
                                             NodeProto &node,
                                             OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Slice, Slice);
