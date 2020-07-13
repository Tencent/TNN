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

#ifndef TNNCONVERTER_SRC_TFLITE_TF_LITE_OP_CONVERTER_H_
#define TNNCONVERTER_SRC_TFLITE_TF_LITE_OP_CONVERTER_H_

#include <map>

#include "onnx_utils.h"
#include "tflite-schema/schema_generated.h"

struct NodeInfo {
    onnx::NodeProto node_proto;
    std::vector<onnx::ValueInfoProto> value_info_list;
    std::vector<onnx::TensorProto> tensor_list;
};

class TFLiteOpConverter {
public:
    virtual void run(NodeInfo& dst_op, const std::unique_ptr<tflite::OperatorT>& tf_lite_op,
                     const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                     const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                     const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set) = 0;
    virtual std::string op_type()                                                               = 0;

    TFLiteOpConverter() {}
    virtual ~TFLiteOpConverter() {}

    friend class TFLiteOpConverterSuit;
};

class TFLiteOpConverterSuit {
public:
    static TFLiteOpConverterSuit* get();
    void insert(TFLiteOpConverter* t, const tflite::BuiltinOperator op_index);
    TFLiteOpConverter* search(const tflite::BuiltinOperator op_index);

    TFLiteOpConverterSuit() {}
    ~TFLiteOpConverterSuit();

private:
    static TFLiteOpConverterSuit* _unique_suit;
    std::map<tflite::BuiltinOperator, TFLiteOpConverter*> _tf_lite_op_converters;
};

template <class T>
class TFLiteOpConverterRegister {
public:
    TFLiteOpConverterRegister(const tflite::BuiltinOperator op_index) {
        T* converter                        = new T;
        TFLiteOpConverterSuit* tf_lite_suit = TFLiteOpConverterSuit::get();
        tf_lite_suit->insert(converter, op_index);
    }

    ~TFLiteOpConverterRegister() {}
};

#define DECLARE_OP_COVERTER(name)                                                                                      \
    class name : public TFLiteOpConverter {                                                                            \
    public:                                                                                                            \
        virtual void run(NodeInfo& dst_op, const std::unique_ptr<tflite::OperatorT>& tf_lite_op,                       \
                         const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,                         \
                         const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,                    \
                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set);                   \
                                                                                                                       \
        name() {}                                                                                                      \
        virtual ~name(){};                                                                                               \
                                                                                                                       \
        virtual std::string op_type();                                                                                 \
    }

#define REGISTER_CONVERTER(name, opType) static TFLiteOpConverterRegister<name> _Convert##opType(opType)

#endif  // TNNCONVERTER_SRC_TFLITE_TF_LITE_OP_CONVERTER_H_
