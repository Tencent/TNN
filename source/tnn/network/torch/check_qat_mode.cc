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
