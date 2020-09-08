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

#ifndef TNNCONVERTER_SRC_TFLITE_TF_LITE_CONVERTER_H_
#define TNNCONVERTER_SRC_TFLITE_TF_LITE_CONVERTER_H_

#include "tflite-schema/schema_generated.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"
#include "utils/model_config.h"

namespace TNN_CONVERTER {
class TFLite2Tnn {
public:
    TFLite2Tnn(std::string model_path);
    TFLite2Tnn(std::string model_path, std::string onnx_path);
    TFLite2Tnn(std::string mode_path, std::string model_name, std::string onnx_path);
    ~TFLite2Tnn(){};
    TNN_NS::Status Convert2Tnn(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource);

private:
    void ReadModel(std::string tf_lite_model_path);
    bool IsQuantized();
    std::string tf_lite_model_name_;
    std::string tf_lite_model_path_;
    std::string onnx_model_path_;
    std::unique_ptr<tflite::ModelT> tf_lite_model_;
};
};  // namespace TNN_CONVERTER

#endif  // TNNCONVERTER_SRC_TFLITE_TF_LITE_CONVERTER_H_
