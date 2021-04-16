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

#include "tnn/interpreter/default_model_interpreter.h"

namespace TNN_NS {

DefaultModelInterpreter::DefaultModelInterpreter() {
    net_structure_ = new NetStructure();
    net_resource_  = new NetResource();
    params_md5_.clear();
}

DefaultModelInterpreter::~DefaultModelInterpreter() {
    if (nullptr != net_structure_)
        delete net_structure_;
    if (nullptr != net_resource_)
        delete net_resource_;
}

NetStructure *DefaultModelInterpreter::GetNetStructure() {
    return net_structure_;
}

NetResource *DefaultModelInterpreter::GetNetResource() {
    return net_resource_;
}

std::vector<std::string> DefaultModelInterpreter::GetParamsMd5() {
    return params_md5_;
}

}  // namespace TNN_NS
