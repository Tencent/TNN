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

#include "check_qat_mode.h"

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

bool inlineCalls(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it++;
    switch (cur->kind()) {
      case prim::CallFunction: {
        AT_ASSERT(cur->input(0)->node()->kind() == prim::Constant);
        auto function_constant = cur->input(0)->node();
        auto fun_type =
            function_constant->output()->type()->expect<FunctionType>();
        if (!fun_type->function()->isGraphFunction()) {
          continue;
        }
        inlineCalls(fun_type->function()->graph()->block());
      } break;
      case prim::CallMethod: {
        const std::string& name = cur->s(attr::name);
        if (auto class_type = cur->input(0)->type()->cast<ClassType>()) {
          Function& function = class_type->getMethod(name);
          if (!function.isGraphFunction()) {
            continue;
          }
          inlineCalls(function.graph()->block());
        }
      } break;
      default: {
        // 1562: aten::fake_quantize_per_tensor_affine
        // 2114 aten::fake_quantize_per_channel_affine
        if(cur->kind() == 1562 || cur->kind() == 2114)
          return true;
        std::cout<<"PengNodeKind:"<<cur->kind().toQualString()<<" Kind:"<<cur->kind()<<std::endl;
        for (auto b : cur->blocks()) {
          inlineCalls(b);
        }
      } break;
    }
  }
  return false;
}

bool CheckQatMode(Graph& graph) {
   return inlineCalls(graph.block());
}

} // namespace jit
} // namespace torch
