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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_ONNX_ONNX_CONVERTER_H_
#define TNN_TOOLS_CONVERTER_SOURCE_ONNX_ONNX_CONVERTER_H_
#include "onnx.pb.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"

namespace TNN_CONVERTER {
class Onnx2Tnn {
public:
    Onnx2Tnn(std::string model_path);
    Onnx2Tnn() = delete;
    ~Onnx2Tnn();
    TNN_NS::Status Converter2Tnn(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource);

private:
    bool ReadModel();
    bool IsQuantized();
    std::string onnx_model_path_;
    std::unique_ptr<onnx::ModelProto> onnx_model_;
};

}  // namespace TNN_CONVERTER

#endif  // TNN_TOOLS_CONVERTER_SOURCE_ONNX_ONNX_CONVERTER_H_
