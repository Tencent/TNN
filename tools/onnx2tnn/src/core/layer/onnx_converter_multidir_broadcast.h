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
#ifndef onnx_converter_multidir_broadcast_hpp_
#define onnx_converter_multidir_broadcast_hpp_
#include <memory>
#include <tuple>

#include "onnx_op_converter.h"
#include "onnx_utility.h"

class OnnxOpConverterMultiBrodcast : public OnnxOpConverter {
public:
    OnnxOpConverterMultiBrodcast(string ignore) : OnnxOpConverter(ignore){};
    virtual ~OnnxOpConverterMultiBrodcast(){};
    virtual string TNNOpType(NodeProto &, OnnxNetInfo &) = 0;
    virtual string TNNLayerParam(NodeProto &, OnnxNetInfo &);
    virtual bool HasLayerResource(NodeProto &node, OnnxNetInfo &net_info);
    virtual int WriteTNNModel(Serializer *, NodeProto &, OnnxNetInfo &);

protected:
    std::tuple<int, std::string> GetWeightInputIndexName(NodeProto &, OnnxNetInfo &);
};

#define DECLARE_MULTI_BROADCASR_OP_CONVERTER(onnx_type)                                                                \
    class OnnxOpConverter##onnx_type : public OnnxOpConverterMultiBrodcast {                                           \
    public:                                                                                                            \
        OnnxOpConverter##onnx_type(string ignore) : OnnxOpConverterMultiBrodcast(ignore){};                            \
        virtual ~OnnxOpConverter##onnx_type(){};                                                                       \
        virtual string TNNOpType(NodeProto &, OnnxNetInfo &net_info);                                                  \
        virtual string TNNLayerParam(NodeProto &, OnnxNetInfo &);                                                      \
        virtual bool HasLayerResource(NodeProto &, OnnxNetInfo &); \
        virtual int WriteTNNModel(Serializer *, NodeProto &, OnnxNetInfo &);                                           \
    }

#define REGISTER_MULTI_BROADCASR_OP_CONVERTER(converter_suffix, onnx_type)                                             \
    REGISTER_OP_CONVERTER(converter_suffix, onnx_type)

#endif
