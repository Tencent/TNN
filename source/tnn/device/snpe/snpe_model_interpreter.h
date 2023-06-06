// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TNN_SOURCE_TNN_DEVICE_SNPE_SNPE_MODEL_INTERPRETER_H_
#define TNN_SOURCE_TNN_DEVICE_SNPE_SNPE_MODEL_INTERPRETER_H_

#include <memory>
#include <vector>

#include "DlContainer/IDlContainer.hpp"

#include "tnn/core/status.h"
#include "tnn/interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

// @brief SNPE model interpreter interpret SNPE model
class SnpeModelInterpreter : public AbstractModelInterpreter {
public:
    SnpeModelInterpreter();

    // @brief virtual destructor
    virtual ~SnpeModelInterpreter();

    // @brief different interpreter has different order param
    virtual Status Interpret(std::vector<std::string>& params);

    // @brief get SNPE container, only in SNPE Model Interpreter
    std::unique_ptr<zdl::DlContainer::IDlContainer>& GetContainer();

private:
    std::unique_ptr<zdl::DlContainer::IDlContainer> container_;
};

}  // namespace TNN_NS
#endif  // TNN_SOURCE_TNN_DEVICE_SNPE_SNPE_MODEL_INTERPRETER_H_
