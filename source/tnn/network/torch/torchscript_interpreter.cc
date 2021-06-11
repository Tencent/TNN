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

#include <memory>

#include "tnn/interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

class TorchScriptInterpreter : public AbstractModelInterpreter {
public:
    TorchScriptInterpreter(){};
    ~TorchScriptInterpreter(){};

    // @brief copy interpreter
    virtual std::shared_ptr<AbstractModelInterpreter> Copy() {
        std::shared_ptr<AbstractModelInterpreter> interp(new TorchScriptInterpreter());
        return interp;
    };

    Status Interpret(std::vector<std::string> &params) {
        return TNN_OK;
    }
};

TypeModelInterpreterRegister<TypeModelInterpreterCreator<TorchScriptInterpreter>> g_torchscript_interpreter_register(
    MODEL_TYPE_TORCHSCRIPT);

}  // namespace TNN_NS
