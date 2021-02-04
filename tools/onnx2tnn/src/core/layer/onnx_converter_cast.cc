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

DECLARE_OP_CONVERTER(Cast);

string OnnxOpConverterCast::TNNOpType(NodeProto &node,
                                           OnnxNetInfo &net_info) {
    return "Cast";
}

string OnnxOpConverterCast::TNNLayerParam(NodeProto &node,
                                               OnnxNetInfo &net_info) {
    const std::string &onnx_op = node.op_type();
    ostringstream layer_param;
    
    //    TensorProto_DataType_UNDEFINED = 0,
    //    TensorProto_DataType_FLOAT = 1,
    //    TensorProto_DataType_UINT8 = 2,
    //    TensorProto_DataType_INT8 = 3,
    //    TensorProto_DataType_UINT16 = 4,
    //    TensorProto_DataType_INT16 = 5,
    //    TensorProto_DataType_INT32 = 6,
    //    TensorProto_DataType_INT64 = 7,
    //    TensorProto_DataType_STRING = 8,
    //    TensorProto_DataType_BOOL = 9,
    //    TensorProto_DataType_FLOAT16 = 10,
    //    TensorProto_DataType_DOUBLE = 11,
    //    TensorProto_DataType_UINT32 = 12,
    //    TensorProto_DataType_UINT64 = 13,
    //    TensorProto_DataType_COMPLEX64 = 14,
    //    TensorProto_DataType_COMPLEX128 = 15,
    //    TensorProto_DataType_BFLOAT16 = 16
    
    //转成common.h里面的DataType值
    int64_t to = get_node_attr_i(node, "to");
    DataType data_type = GetTnnDataTypeFromOnnx(to);
    layer_param << data_type << " ";

    return layer_param.str();
}

bool OnnxOpConverterCast::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    return false;
}

int OnnxOpConverterCast::WriteTNNModel(Serializer *net_writer,
                                            NodeProto &node,
                                            OnnxNetInfo &net_info) {
    //有权值写入的返回1， 没有的返回0
    return 0;
}

REGISTER_OP_CONVERTER(Cast, Cast);
