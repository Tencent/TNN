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

#include "include/tnn/core/tnn.h"
#include "onnx/onnx_converter.h"
#include "optimizer/tnn_optimizer.h"
#include "resource/resource_convert.h"
#include "runtime/tnn_runtime.h"
#include "tflite/tflite_converter.h"
#include "tnn/interpreter/abstract_model_interpreter.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"
#include "utils/command.h"
#include "utils/flags.h"
#include "utils/generate_model.h"
#include "utils/model_config.h"

namespace TNN_CONVERTER {
int Run(int argc, char* argv[]) {
    ParseCommandLine(argc, argv);

    auto interpreter =
        std::shared_ptr<TNN_NS::AbstractModelInterpreter>(TNN_NS::CreateModelInterpreter(TNN_NS::MODEL_TYPE_TNN));
    TNN_NS::NetStructure& net_structure =
        *(dynamic_cast<TNN_NS::DefaultModelInterpreter*>(interpreter.get())->GetNetStructure());
    TNN_NS::NetResource& net_resource =
        *(dynamic_cast<TNN_NS::DefaultModelInterpreter*>(interpreter.get())->GetNetResource());
    ModelConfig model_config(FLAGS_mt, FLAGS_mp, FLAGS_od);
    TNN_NS::Status status;
    if (model_config.model_type_ == TNN_CONVERTER::MODEL_TYPE_TF_LITE) {
        TFLite2Tnn tf_lite_2_tnn(model_config.model_path_);
        status = tf_lite_2_tnn.Convert2Tnn(net_structure, net_resource);
    } else if (model_config.model_type_ == TNN_CONVERTER::MODEL_TYPE_ONNX) {
        Onnx2Tnn onnx_2_tnn(model_config.model_path_);
        status = onnx_2_tnn.Converter2Tnn(net_structure, net_resource);
    }
    if (status != TNN_NS::TNN_CONVERT_OK) {
        LOGE("Converter: converter %s failed!\n", FLAGS_mp.c_str());
        return status;
    }
    // TODO optimize the model
    // prefer optimize
    TnnOptimizer tnn_optimizer;
    status = tnn_optimizer.PreOptimize(net_structure, net_resource);
    if (status != TNN_NS::TNN_CONVERT_OK) {
        LOGE("Converter: optimize %s failed!\n", FLAGS_mp.c_str());
        return status;
    }
    // tnn run time
    TnnRuntime tnn_runtime;
    status = tnn_runtime.ConstantFolding(interpreter);
    if (status != TNN_NS::TNN_OK) {
        LOGE("Converter: tnn runtime failed\n");
        return status;
    }
    // post optimize
    status = tnn_optimizer.PostOptimize(net_structure, net_resource);
    if (status != TNN_NS::TNN_CONVERT_OK) {
        LOGE("Converter: optimize %s failed!\n", FLAGS_mp.c_str());
        return status;
    }
    // convert weight data type
    ResourceConvert resource_manager;
    if (FLAGS_half) {
        resource_manager.SetResourceConvertType(RESOURCE_CONVERT_HALF);
    }
    resource_manager.converter(net_structure, net_resource);
    // wright the model
    std::string file_name = GetFileName(model_config.model_path_);
    status                = GenerateModel(net_structure, net_resource, model_config.output_dir_, file_name);
    if (status != TNN_NS::TNN_CONVERT_OK) {
        LOGE("Converter: generate tnn model failed!\n");
        return status;
    }
    return 0;
}

}  // namespace TNN_CONVERTER
int main(int argc, char* argv[]) {
    return TNN_CONVERTER::Run(argc, argv);
}
