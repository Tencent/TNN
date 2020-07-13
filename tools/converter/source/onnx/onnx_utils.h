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

#ifndef TNNCONVERTER_SRC_TFLITE_ONNX_UTILS_H_
#define TNNCONVERTER_SRC_TFLITE_ONNX_UTILS_H_

#include "onnx.pb.h"

onnx::ValueInfoProto MakeValueInfo(const std::string& name, const std::vector<int>& data_shape,
                                   const onnx::TensorProto::DataType& data_type);

template <class T>
onnx::TensorProto MakeTensor(const std::string& name, const std::vector<T>& v, onnx::TensorProto_DataType& data_type);

onnx::AttributeProto MakeAttribute(const std::string& name, const std::vector<int>& vals);
onnx::AttributeProto MakeAttribute(const std::string& name, const std::vector<float>& vals);
onnx::AttributeProto MakeAttribute(const std::string& name, int64_t val);
onnx::AttributeProto MakeAttribute(const std::string& name, const std::string& val);
onnx::AttributeProto MakeAttribute(const std::string& name, onnx::TensorProto& val);

onnx::NodeProto MakeNode(const std::string& type, const std::vector<std::string>& inputs,
                         const std::vector<std::string>& outputs, const std::vector<onnx::AttributeProto>& attributes,
                         const std::string& name = "");
inline onnx::NodeProto MakeNode(const std::string& type, const std::vector<std::string>& inputs,
                                const std::vector<std::string>& outputs, const std::string& name = "") {
    return MakeNode(type, inputs, outputs, {}, name);
}

#endif  // TNNCONVERTER_SRC_TFLITE_ONNX_UTILS_H_