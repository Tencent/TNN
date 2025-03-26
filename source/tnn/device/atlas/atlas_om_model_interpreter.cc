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

#include <fstream>
#include "tnn/device/atlas/atlas_om_model_interpreter.h"
#include "tnn/device/atlas/atlas_utils.h"
#include "tnn/utils/split_utils.h"

namespace TNN_NS {

AtlasOMModelInterpreter::AtlasOMModelInterpreter() {}

AtlasOMModelInterpreter::~AtlasOMModelInterpreter() {}

Status AtlasOMModelInterpreter::Interpret(std::vector<std::string> &params) {
    // OM Model Load API only support LOAD model directly ONTO device (Card)
    // So the real model interpret path is in AtlasNetwork instead.
    
    // The only thing we need to do here is to store om_string locally,
    // we USE MOVE here to save memory for large OM model.
    this->om_str_ = std::move(params[0]);
    //this->om_str_ = params[0];

    return TNN_OK;
}

std::string& AtlasOMModelInterpreter::GetOmString() {
    return this->om_str_;
}

TypeModelInterpreterRegister<TypeModelInterpreterCreator<AtlasOMModelInterpreter>> g_atlas_model_interpreter_register(
    MODEL_TYPE_ATLAS);

}  // namespace TNN_NS
