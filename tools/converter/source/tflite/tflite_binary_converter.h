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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_TFLITE_TFLITE_BINARY_CONVERTER_H_
#define TNN_TOOLS_CONVERTER_SOURCE_TFLITE_TFLITE_BINARY_CONVERTER_H_
#include "tflite-schema/schema_generated.h"
#include "tflite_op_converter.h"

namespace TNN_CONVERTER {

class TFLiteBinaryConverter : public TFLiteOpConverter {
public:
    TFLiteBinaryConverter() : TFLiteOpConverter(){};
    virtual ~TFLiteBinaryConverter(){};
    virtual TNN_NS::Status exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                                const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                                const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                                const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                                const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,
                                bool quantizedModel);
    virtual std::string TNNOpType(bool quantimized_mode) = 0;
};

#define DECLARE_BINARY_CONVERTER(tf_lite_type)                                                                         \
    class TFLite##tf_lite_type##Converter : public TFLiteBinaryConverter {                                             \
    public:                                                                                                            \
        TFLite##tf_lite_type##Converter() : TFLiteBinaryConverter(){};                                                 \
        virtual ~TFLite##tf_lite_type##Converter(){};                                                                  \
        virtual TNN_NS::Status exec(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,            \
                                    const std::unique_ptr<tflite::OperatorT>& tfliteOp,                                \
                                    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,                \
                                    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,            \
                                    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet,            \
                                    bool quantizedModel);                                                              \
        virtual std::string TNNOpType(bool quantizedModel);                                                            \
    }  // namespace TNN_CONVERTER

}  // namespace TNN_CONVERTER
//
#endif  // TNN_TOOLS_CONVERTER_SOURCE_TFLITE_TFLITE_BINARY_CONVERTER_H_