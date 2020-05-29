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

DECLARE_OP_CONVERTER(Const);

string OnnxOpConverterConst::TNNOpType(NodeProto &node,
                                           OnnxNetInfo &net_info) {
    return "Const";
}

string OnnxOpConverterConst::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    const int dim_size = net_info.is_3D_model ? 5 : 4;
    std::vector<int> dims;
    if (dim_size == 5) {
        dims = {1, 1, 1, 1, 1};
    } else {
        dims = {1, 1, 1, 1};
    }
    
    auto name = node.output(0);
    auto shape = net_info.weights_shape_map[name];
    onnx::TensorProto tensor = get_node_attr_tensor(node, "value");
    int data_size = get_tensor_proto_data_size(tensor);
    for (int i = tensor.dims_size()-1, j=dim_size-1; i>=0 && j>=0; i--, j--) {
        dims[j] = tensor.dims(i);
    }
    
    ostringstream layer_param;
    
    for (int i=0; i<dim_size; i++) {
        layer_param << dims[i] << " ";
    }
    return layer_param.str();
    
//    int i = start_id_;
//    int n1 = atoi(param[i].c_str());
//    ++i;
//    for (int j = 0; j < 4 - n1; j++) {
//        _shape.add_dim(1);
//    }
//    for(;n1 > 0; i++, n1--) {
//        _shape.add_dim(atoi(param[i].c_str()));
//    }
}

int OnnxOpConverterConst::WriteTNNModel(serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    onnx::TensorProto tensor = get_node_attr_tensor(node, "value");
    WriteTensorData(tensor, net_writer, net_info.data_type);
    return 1;
}

REGISTER_OP_CONVERTER(Const, Constant);

