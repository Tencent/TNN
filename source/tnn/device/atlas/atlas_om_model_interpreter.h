// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_OM_MODEL_INTERPRETER_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_OM_MODEL_INTERPRETER_H_

#include <climits>
#include <memory>
#include <map>
#include <mutex>
#include <vector>
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/device/atlas/atlas_common_types.h"
#include "tnn/interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

// @brief Atlas OM model interpreter that interprets Atlas OM Model
class AtlasOMModelInterpreter : public AbstractModelInterpreter {
public:
    AtlasOMModelInterpreter();

    // @brief virtual destructor
    virtual ~AtlasOMModelInterpreter();

    // @brief different interpreter has different order param
    virtual Status Interpret(std::vector<std::string> &params);

    // @brief get model om string
    std::string& GetOmString();
    
private:
    std::string om_str_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_OM_MODEL_INTERPRETER_H_
