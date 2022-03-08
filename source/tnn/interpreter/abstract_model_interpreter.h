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

#ifndef TNN_SOURCE_TNN_INTERPRETER_ABSTRACT_MODEL_INTERPRETER_H_
#define TNN_SOURCE_TNN_INTERPRETER_ABSTRACT_MODEL_INTERPRETER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tnn/core/common.h"
#include "tnn/core/status.h"

namespace TNN_NS {

// @brief AbstraceModelInterpreter define common interface, different
// interpreter different style model.
class AbstractModelInterpreter {
public:
    // @brief virtual destructor
    virtual ~AbstractModelInterpreter(){};

    // @brief different interpreter has different order param
    virtual Status Interpret(std::vector<std::string>& params) = 0;

    // @brief interpret extra config, such as conv winograd for specific conv layer
    virtual Status InterpretConfig(std::map<std::string, std::string>& config_map) {
        return TNN_OK;
    };

    // @brief copy interpreter
    virtual std::shared_ptr<AbstractModelInterpreter> Copy() {
        return nullptr;
    };
};

// @brief ModelInterpreterCreator define model interpreter creator interface
class ModelInterpreterCreator {
public:
    virtual ~ModelInterpreterCreator(){};
    virtual AbstractModelInterpreter* CreateModelInterpreter() = 0;
};

// @brief TypeModelInterpreterCreator create different type model interpreter
template <typename T>
class TypeModelInterpreterCreator : public ModelInterpreterCreator {
    virtual AbstractModelInterpreter* CreateModelInterpreter() {
        return new T();
    }
};

//@brief TypeModelInterpreterCreator register map
std::map<ModelType, std::shared_ptr<ModelInterpreterCreator>>& GetGlobalModelInterpreterCreatorMap();

//@brief TypeModelInterpreterRegister register TypeModelInterpreterCreator
template <typename T>
class TypeModelInterpreterRegister {
public:
    explicit TypeModelInterpreterRegister(ModelType type) {
        GetGlobalModelInterpreterCreatorMap()[type] = std::shared_ptr<T>(new T());
    }
};

AbstractModelInterpreter* CreateModelInterpreter(ModelType type);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_ABSTRACT_MODEL_INTERPRETER_H_
