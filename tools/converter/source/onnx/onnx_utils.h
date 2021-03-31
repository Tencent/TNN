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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_ONNX_ONNX_UTILS_H_
#define TNN_TOOLS_CONVERTER_SOURCE_ONNX_ONNX_UTILS_H_

#include <cassert>
#include <vector>

#include "onnx.pb.h"
#include "onnx_proxy_graph.h"
#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/raw_buffer.h"

namespace TNN_CONVERTER {

TNN_NS::DimsVector ConvertTensorShapeProtoToDimsVector(onnx::TensorShapeProto tensor_shape_proto);

onnx::AttributeProto_AttributeType GetAttributeType(const char* basic_type_name);

int GetAttributeInt(const onnx::NodeProto& node, const std::string& name, int default_value);

std::vector<int32_t> GetAttributeIntVector(const onnx::NodeProto& node, const std::string& name);

float GetAttributeFloat(const onnx::NodeProto& node, const std::string& name, float default_value);

float GetAttributeFloat(const onnx::NodeProto& node, const std::string& name, int location, float default_value,
                        std::map<std::string, const onnx::TensorProto*> proxy_initializers_map);

std::string GetAttributeString(const onnx::NodeProto& node, const std::string& name, std::string default_value);

std::vector<std::string> GetAttributeStringVector(const onnx::NodeProto& node, const std::string& name);

std::vector<std::string> SplitString(std::string& s, const std::string& c);

std::vector<uint8_t> GetAttributeUInt8Vector(const onnx::NodeProto& node, const std::string& name);

std::vector<int8_t> Asymmetric2Symmetric(std::vector<uint8_t>& raw_value, uint8_t zero_point);

onnx::TensorProto GetAttributeTensor(const onnx::NodeProto& node, const char* key);

const float* GetTensorProtoData(const onnx::TensorProto& tp);

int GetTensorProtoDataSize(const onnx::TensorProto& tp);

void* GetDataFromTensor(const onnx::TensorProto& tensor, onnx::TensorProto_DataType data_type);

void CreateRawBufferFromConstant(const onnx::NodeProto& constant_node, TNN_NS::RawBuffer** raw_buffer);

std::vector<int64_t> GetAttributeInt64Vector(const onnx::NodeProto& node, const std::string& name);

std::vector<int64_t> GetAttributeInt64Vector(const onnx::NodeProto& node, const std::string& name,
                                             std::map<std::string, const onnx::TensorProto*>& proxy_initializers_map,
                                             int location);

const onnx::TensorProto* GetTensorFromConstantNode(const onnx::NodeProto& constant_node);

void CreateRawBufferFromTensor(const onnx::TensorProto& tensor, TNN_NS::RawBuffer** raw_buffer);

TNN_NS::DimsVector CreateDimsVectorFromTensor(const onnx::TensorProto& tensor);

int TensorProtoDataType2TnnDataType(int data_type);

template <typename T>
bool OHWI2OIHW(T* src, T* dst, int CO, int KH, int KW, int CI) {
    ASSERT(CO > 0);
    ASSERT(KH > 0);
    ASSERT(KW > 0);
    ASSERT(CI > 0);
    ASSERT(src != nullptr);
    for (int co = 0; co < CO; ++co) {
        for (int ci = 0; ci < CI; ++ci) {
            for (int h = 0; h < KH; ++h) {
                for (int w = 0; w < KW; ++w) {
                    dst[(co * CI + ci) * KH * KW + h * KW + w] = src[(co * KH + h) * KW * CI + w * CI + ci];
                }
            }
        }
    }
    return true;
}

TNN_NS::DataType GetTnnDataTypeFromOnnx(const onnx::TensorProto_DataType& onnx_type);

template <class T>
std::vector<T> GetTensorProtoDataVector(const onnx::TensorProto& tp);

TNN_NS::Status GetWeightInputIndexName(int& weight_input_index, std::string& weight_name, const onnx::NodeProto& node,
                                       std::map<std::string, const onnx::TensorProto*> proxy_initializers_map,
                                       std::map<std::string, std::shared_ptr<OnnxProxyNode>>& proxy_nodes);

}  // namespace TNN_CONVERTER

#endif  // TNN_TOOLS_CONVERTER_SOURCE_ONNX_ONNX_UTILS_H_
