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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_TORCH_TORCH_CONVERTER_H_
#define TNN_TOOLS_CONVERTER_SOURCE_TORCH_TORCH_CONVERTER_H_

#include <torch/script.h>

#include <memory>

#include "tnn/core/status.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"
#include "utils/model_config.h"

namespace TNN_CONVERTER {
class Torch2Tnn {
public:
    Torch2Tnn(const std::string& model_path);
    Torch2Tnn(const std::string& model_path, const std::string& onnx_path);
    Torch2Tnn(const std::string& mode_path, const std::string& model_name, const std::string& onnx_path);
    ~Torch2Tnn(){};
    TNN_NS::Status Convert2Tnn(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource);

private:
    void ReadModel(std::string torch_model_path);
    bool IsQuantized();
    std::string torch_model_name_;
    std::string torch_model_path_;
    std::string onnx_model_path_;
    torch::jit::Module torch_model_;
    std::shared_ptr<torch::jit::Graph> torch_graph_;
};
};  // namespace TNN_CONVERTER

#endif  // TNN_TOOLS_CONVERTER_SOURCE_TORCH_TORCH_CONVERTER_H_
