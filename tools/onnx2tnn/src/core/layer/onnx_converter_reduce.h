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

#ifndef ONNX2TNN_CORE_LAYER_ONNX_CONVERTER_REDUCE_H_
#define ONNX2TNN_CORE_LAYER_ONNX_CONVERTER_REDUCE_H_
#include "onnx_op_converter.h"
#include "onnx_utility.h"

class OnnxConverterReduce : public OnnxOpConverter {
public:
    OnnxConverterReduce(string tnn_type, string onnx_type)
        : OnnxOpConverter(onnx_type) {
        tnn_type_ = tnn_type;
    };
    virtual ~OnnxConverterReduce(){};
    virtual string TNNOpType(NodeProto &, OnnxNetInfo &) {
        return tnn_type_;
    };
    string TNNLayerParam(NodeProto &, OnnxNetInfo &);
    virtual int WriteTNNModel(Serializer *, NodeProto &, OnnxNetInfo &) {
        return 0;
    };

private:
    string tnn_type_;
};

#define REGISTER_OP_CONVERTER_REDUCE(tnn_type, onnx_type)                      \
    OnnxOpConverterRegister<OnnxConverterReduce>                               \
        g_converter_##tnn_type_##onnx_type(#tnn_type, #onnx_type)

#endif  // ONNX2TNN_CORE_LAYER_ONNX_CONVERTER_REDUCE_H_
