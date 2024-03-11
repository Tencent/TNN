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

#include "tnn/device/atlas/torchaie/torchaie_interpreter.h"
#include "torch_aie.h"

namespace TNN_NS {

Status TorchAieInterpreter::Interpret(std::vector<std::string> &params) {
    module_path_ = params[0];
    return TNN_OK;
}

std::shared_ptr<torch::jit::Module> TorchAieInterpreter::GetModule(bool load_cpu) {
    if (aie_module_.get() == nullptr) {
        try {
            if (load_cpu) {
                aie_module_ = std::make_shared<torch::jit::Module>(torch::jit::load(module_path_, torch::kCPU));
            }
            else {
                aie_module_ = std::make_shared<torch::jit::Module>(torch::jit::load(module_path_));
            }
        } catch(const std::exception& e) {
            std::cout << e.what() << '\n';
        }
    }

    return aie_module_;
}

TypeModelInterpreterRegister<TypeModelInterpreterCreator<TorchAieInterpreter>> g_torchaie_interpreter_register(
    MODEL_TYPE_TORCHAIE);

}  // namespace TNN_NS
