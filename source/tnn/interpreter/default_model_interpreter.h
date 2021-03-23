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

#ifndef TNN_SOURCE_TNN_INTERPRETER_DEFAULT_MODEL_INTERPRETER_H_
#define TNN_SOURCE_TNN_INTERPRETER_DEFAULT_MODEL_INTERPRETER_H_

#include "tnn/core/status.h"
#include "tnn/interpreter/abstract_model_interpreter.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"

namespace TNN_NS {

// @brief DefaultModelInterpreter define common interface for rpn model,
// different interpreter different style model.
class DefaultModelInterpreter : public AbstractModelInterpreter {
public:
    // @brief default constructor
    DefaultModelInterpreter();

    // @brief virtual destructor
    virtual ~DefaultModelInterpreter() = 0;

    // @brief different interpreter has different order param
    virtual Status Interpret(std::vector<std::string> &params) = 0;

    // @brief GetNetStruture return network build info
    virtual NetStructure *GetNetStructure();

    // @brief GetNetResource return network weights data
    virtual NetResource *GetNetResource();

    //@brief GetParamsMd5 return md5 string of params string
    std::vector<std::string> GetParamsMd5();

protected:
    std::vector<std::string> params_md5_;
    NetStructure *net_structure_;
    NetResource *net_resource_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_DEFAULT_MODEL_INTERPRETER_H_
