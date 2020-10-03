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

#include "tnn_runtime.h"

#include "include/tnn/core/common.h"
#include "include/tnn/core/instance.h"

namespace TNN_CONVERTER {

TnnRuntime::TnnRuntime() {
    // initial network config
    network_config_.network_type = TNN_NS::NETWORK_TYPE_DEFAULT;
    network_config_.device_type  = TNN_NS::DEVICE_NAIVE;
    network_config_.precision    = TNN_NS::PRECISION_AUTO;
    network_config_.library_path = {};
    // initial model config
    model_config_.model_type = TNN_NS::MODEL_TYPE_TNN;
    // fake mode config params
    model_config_.params = {};
}
TnnRuntime::~TnnRuntime() {
    // do nothing
}

TNN_NS::Status TnnRuntime::run(std::shared_ptr<TNN_NS::AbstractModelInterpreter> interpreter) {
    // create input shape map
    TNN_NS::DefaultModelInterpreter* tnn_interpreter =
        (dynamic_cast<TNN_NS::DefaultModelInterpreter*>(interpreter.get()));
    TNN_NS::InputShapesMap& input_shapes_map = tnn_interpreter->GetNetStructure()->inputs_shape_map;
    auto instance                            = std::make_shared<TNN_NS::Instance>(network_config_, model_config_);
    auto status                              = instance->Init(interpreter, input_shapes_map);
    instance->Forward();

    return TNN_NS::TNN_OK;
}

//TNN_NS::Status TnnRuntime::run(TNN_NS::DefaultModelInterpreter* interpreter) {
//    TNN_NS::InputShapesMap input_shapes_map = interpreter->GetNetStructure()->inputs_shape_map;
//    auto instance = std::make_shared<TNN_NS::Instance>(network_config_, model_config_);
//    auto status =  instance->Init(std::shared_ptr<TNN_NS::AbstractModelInterpreter>(interpreter), input_shapes_map);
//}

}  // namespace TNN_CONVERTER