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

DECLARE_OP_CONVERTER(Reshape);

string OnnxOpConverterReshape::TNNOpType(NodeProto &node,
                                              OnnxNetInfo &net_info) {
    return "Reshape";
}

string OnnxOpConverterReshape::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    if (node.input_size() == 1 || node.input_size() == 2) {
        const std::string& input_name = node.input(0);
        
        // skip weight reshape
        if (net_info.weights_map.find(input_name) != net_info.weights_map.end()) {
            DLog("reshape of weights is not supported, input0:%s out_node:%s\n", node.input(0).c_str(), node.output(0).c_str());
            assert(0);
        }
    }
    
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;

    int start_axis          = 0;
    int num_axis            = net_info.is_3D_model ? 5 : 4;
    int top_blob_shape_size = net_info.is_3D_model ? 5 : 4;
    layer_param << start_axis << " " << num_axis << " "
    << top_blob_shape_size << " ";
    
    const onnx::TensorProto& shape_tp = net_info.weights_map[node.input(1)];
    const int64_t* shape_data =
    (const int64_t*)get_tensor_proto_data(shape_tp);
    int shape_dim = get_tensor_proto_data_size(shape_tp);
    
    std::vector<int> output_shape;
    for (int ii=0; ii<top_blob_shape_size; ii++) {
        if (ii < shape_dim) {
            output_shape.push_back((int)shape_data[ii]);
        } else {
            output_shape.push_back(1);
        }
    }

    int param_ii = 0;
    //兼容gpu上SbatchSize>1的情况
    { 
        layer_param << 0 << " ";
        param_ii++;
    }
    
    for(;param_ii < top_blob_shape_size;param_ii++) {
        layer_param << output_shape[param_ii] << " ";
    }
    
    return layer_param.str();
}

int OnnxOpConverterReshape::WriteTNNModel(serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Reshape, Reshape);
