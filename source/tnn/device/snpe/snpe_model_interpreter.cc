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


#include <fstream>

#include "tnn/device/snpe/snpe_model_interpreter.h"

namespace TNN_NS {

SnpeModelInterpreter::SnpeModelInterpreter() {}

SnpeModelInterpreter::~SnpeModelInterpreter() {}

Status SnpeModelInterpreter::Interpret(std::vector<std::string>& params) {
    std::string dlc_content = params[0];
    std::ifstream dlc_file(dlc_content);
    if (!dlc_file) {
        LOGE("SnpeModelInterpreter: Invalied dlc file path!\n");
        return TNNERR_INVALID_MODEL;
    }

    container_ = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(dlc_content.c_str()));
    if (container_ == nullptr) {
        LOGE("SnpeModelInterpreter: Load dlc file failed!\n");
        return TNNERR_INVALID_MODEL;
    }

    return TNN_OK;
}

std::unique_ptr<zdl::DlContainer::IDlContainer>& SnpeModelInterpreter::GetContainer() {
    return container_;
}

TypeModelInterpreterRegister<TypeModelInterpreterCreator<SnpeModelInterpreter>>
    g_snpe_model_interpreter_register(MODEL_TYPE_SNPE);

}  // namespace TNN_NS
