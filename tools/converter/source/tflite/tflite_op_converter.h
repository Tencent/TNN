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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_TFLITE_TFLITE_OP_CONVERTER_H_
#define TNN_TOOLS_CONVERTER_SOURCE_TFLITE_TFLITE_OP_CONVERTER_H_

#include "tflite/tflite-schema/schema_generated.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"

namespace TNN_CONVERTER {

class TFLiteOpConverter {
public:
    TFLiteOpConverter()          = default;
    virtual ~TFLiteOpConverter() = default;

    virtual TNN_NS::Status exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,
                                const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,
                                const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,
                                bool quantized_model)                                      = 0;
    virtual std::string TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model)   = 0;
    virtual tflite::ActivationFunctionType ActivationType(const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,
                                                          tflite::BuiltinOperator op_code) = 0;
    TNN_NS::Status SeparateActivation(TNN_NS::NetStructure& net_structure,
                                      tflite::ActivationFunctionType activation_function_type);
    void InsertBlobs(TNN_NS::NetStructure& net_structure);

protected:
    std::string tflite_op_type_;
};

class TFLiteOpConverterManager {
public:
    static TFLiteOpConverterManager* get();
    void insert(const tflite::BuiltinOperator op_index, TFLiteOpConverter* tflite_op_converter);
    TFLiteOpConverter* search(const tflite::BuiltinOperator op_index);
    TFLiteOpConverterManager(){};
    ~TFLiteOpConverterManager();

private:
    static TFLiteOpConverterManager* tf_lite_op_converter_manager_;
    std::map<tflite::BuiltinOperator, TFLiteOpConverter*> tf_lite_op_converter_map_;
};

template <class T>
class TFLiteOpConverterRegister {
public:
    explicit TFLiteOpConverterRegister(const tflite::BuiltinOperator op_index) {
        T* converter                                           = new T;
        TFLiteOpConverterManager* tf_lite_op_converter_manager = TFLiteOpConverterManager::get();
        tf_lite_op_converter_manager->insert(op_index, converter);
    };
    ~TFLiteOpConverterRegister(){};
};

#define DECLARE_OP_CONVERTER(tf_lite_type)                                                                             \
    class TFLite##tf_lite_type##Converter : public TFLiteOpConverter {                                                 \
    public:                                                                                                            \
        TFLite##tf_lite_type##Converter() {}                                                                           \
        virtual ~TFLite##tf_lite_type##Converter() {}                                                                  \
        virtual TNN_NS::Status exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,            \
                                    const std::unique_ptr<tflite::OperatorT>& tf_lite_operator,                        \
                                    const std::vector<std::unique_ptr<tflite::TensorT>>& tf_lite_tensors,              \
                                    const std::vector<std::unique_ptr<tflite::BufferT>>& tf_lite_model_buffer,         \
                                    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tf_lite_op_set,         \
                                    bool quantized_model);                                                             \
        virtual std::string TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model);                          \
        virtual tflite::ActivationFunctionType ActivationType(                                                         \
            const std::unique_ptr<tflite::OperatorT>& tf_lite_operator, tflite::BuiltinOperator op_code);              \
    }  // namespace TNN_CONVERTER

#define REGISTER_CONVERTER(converter_suffix, tf_lite_type)                                                             \
    TFLiteOpConverterRegister<TFLite##converter_suffix##Converter> g_converter_##tf_lite_type##_(tf_lite_type)
};  // namespace TNN_CONVERTER

#endif  // TNN_TOOLS_CONVERTER_SOURCE_TFLITE_TFLITE_OP_CONVERTER_H_
