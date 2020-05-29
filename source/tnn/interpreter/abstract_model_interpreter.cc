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

#include "tnn/interpreter/abstract_model_interpreter.h"
#include <mutex>

namespace TNN_NS {

std::map<ModelType, std::shared_ptr<ModelInterpreterCreator>> &GetGlobalModelInterpreterCreatorMap() {
    static std::once_flag once;
    static std::shared_ptr<std::map<ModelType, std::shared_ptr<ModelInterpreterCreator>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<ModelType, std::shared_ptr<ModelInterpreterCreator>>); });
    return *creators;
}

AbstractModelInterpreter *CreateModelInterpreter(ModelType type) {
    AbstractModelInterpreter *interpreter = NULL;
    auto &creater_map                     = GetGlobalModelInterpreterCreatorMap();
    if (creater_map.count(type) > 0) {
        interpreter = creater_map[type]->CreateModelInterpreter();
    }
    return interpreter;
}

}  // namespace TNN_NS
