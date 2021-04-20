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

#ifndef TNN_TOOLS_CONVERTER_SOURCE_RUNTIME_TNN_RUNTIME_H_
#define TNN_TOOLS_CONVERTER_SOURCE_RUNTIME_TNN_RUNTIME_H_
#include "include/tnn/core/common.h"
#include "include/tnn/core/mat.h"
#include "include/tnn/core/status.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/interpreter/net_structure.h"
#include "tnn/utils/blob_converter.h"

namespace TNN_CONVERTER {

class TnnRuntime {
public:
    TnnRuntime();
    ~TnnRuntime();
    TNN_NS::Status ConstantFolding(const std::shared_ptr<TNN_NS::AbstractModelInterpreter> interpreter);
    TNN_NS::Status AlignModel(const std::shared_ptr<TNN_NS::AbstractModelInterpreter> interpreter);

private:
    TNN_NS::MatMap CreateBlobMatMap(TNN_NS::BlobMap& blob_map, int format_type);
    void InitInputMatMap(TNN_NS::MatMap& mat_map);
    std::map<std::string, std::shared_ptr<TNN_NS::BlobConverter>> CreateBlobConverterMap(TNN_NS::BlobMap& blob_map);
    std::map<std::string, TNN_NS::MatConvertParam> CreateConvertParamMap(TNN_NS::MatMap& mat_map);
    TNN_NS::NetworkConfig network_config_;
    TNN_NS::ModelConfig model_config_;
};
}  // namespace TNN_CONVERTER
#endif  // TNN_TOOLS_CONVERTER_SOURCE_RUNTIME_TNN_RUNTIME_H_
