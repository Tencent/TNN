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

#ifndef TNN_SOURCE_TNN_NETWORK_TNNTORCH_ATTRIBUTE_PROPAGATOR_H_
#define TNN_SOURCE_TNN_NETWORK_TNNTORCH_ATTRIBUTE_PROPAGATOR_H_

#include <torch/csrc/jit/passes/freeze_module.h>

#include <torch/csrc/jit/jit_log.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

#include <stack>

namespace torch {
namespace jit {

namespace {
class AttributePropagator {
 public:
  AttributePropagator(
      Module& module,
      std::vector<std::string> preservedAttrs = std::vector<std::string>(),
      bool freezeInterfaces = true,
      bool preserveParameters = false)
      : module_(module),
        freezeInterfaces_(freezeInterfaces),
        preserveParameters_(preserveParameters) {
    // Currently only top level attributes and functions can  be preserved
    // explicitly.
    auto checkName = [this](std::string& name) {
      if (module_.hasattr(name)) {
        auto attr = module_.attr(name);

        // Freezing client wants to presever this submodule. When cleaning
        // the frozen module, make sure it will be preserved entirely.
        if (attr.isModule()) {
          preservedSubModule_.insert(attr.toModule()._ivalue());
        }
        insertMutableAttr(name, attr, module_._ivalue());
        return true;
      }

      for (auto& fn : module_.type()->methods()) {
        if (fn->name() == name) {
          preservedMethods_.insert(fn);
          return true;
        }
      }
      return false;
    };

    // forward is preserved by default, but
    // not all modules have a forward function defined
    if (module_.find_method("forward")) {
      auto method = module_.get_method("forward");
      preservedMethods_.insert(&method.function());
    }

    for (auto name : preservedAttrs) {
      TORCH_CHECK(checkName(name), "Unknown name: " + name);
    }
  }

 private:
  // findConstantAttr function locates the sub Module where attributes are
  // defined. The algorithm chases getAttr chains to locate the submodules.
  // For example:
  // module M {
  //   attributes {
  //     A = <SubModule at ...>
  //   }
  //   ...
  //   %A = prim::GetAttr[name="A"](%self)
  //   ...
  //   %B = prim::GetAttr[name="B"](%A)
  //   ...
  //   %weight = prim::GetAttr[name="scale"](%B)
  //   ...
  //   submodules {
  //     module SubModule {
  //       attributes {
  //          B = <SubModule2 at ...>
  //       }
  //       submodules {
  //         module SubModule2 {
  //            attributes {
  //               scale = 2
  //            }
  //         }
  //       }
  //     }
  //   }
  //
  // findConstantAttr(%B, "scale", M)  returns true because there are no
  // explicit SetAttr that modifies %B. attrModule points to the module where
  // attribute lives (in this example it is <SubModule2 at ...>).
  //
  // Note inplace mutations to attributes are checked later using alias
  // analysis.
  //
  // We can use a more efficient algorithm to hash each constant GetAttr to its
  // corresponding value. Based on initial test on resnet50 and other torch
  // vision tests. GetAttrs are not too frequent so it is ok to chase GetAttr
  // chain to retrieve their values.
  bool findConstantAttr(
      Value* input,
      std::string& name,
      Module& attrModule,
      std::shared_ptr<Graph>& graph) {
    if (!input->type()->cast<InterfaceType>() &&
        !input->type()->expectRef<ClassType>().is_module()) {
      return false;
    }

    Node* node = input->node();
    names_.clear();
    while (!(node->outputs()[0]->type() == graph->inputs()[0]->type())) {
      if (node->kind() == prim::GetAttr) {
        names_.push_front(node->s(attr::name));
        node = node->inputs()[0]->node();
      } else {
        return false;
      }
    }

    for (auto& moduleName : names_) {
      if (preservedAttrs_.count(attrModule.attr(moduleName))) {
        return false;
      }
      attrModule = attrModule.attr(moduleName).toModule();
    }

    auto attr = attrModule.attr(name);
    /*
    if (!AliasDb::isMutableType(attr.type())) {
      auto it = preservedScalarAttrs_.find(attrModule._ivalue());
      return it == preservedScalarAttrs_.end() || !it->second.count(name);
    }
    */
    if (preservedAttrs_.count(attr)) {
      return false;
    }
    if (!attr.type()->cast<ClassType>()) {
      for (auto& ivalue : preservedAttrs_) {
        if (!ivalue.isObject() && ivalue.overlaps(attr)) {
          return false;
        }
      }
    }
    return true;
  }

  void insertMutableAttr(
      const std::string& name,
      const IValue& attr,
      const ModulePtr& attrModule) {
    /* 
    if (AliasDb::isMutableType(attr.type())) {
      preservedAttrs_.insert(attr);
    } else {
      preservedScalarAttrs_[attrModule].insert(name);
    }
    */
  }
public:
  IValue overrideGradient(IValue attr) {
    if (attr.isTensor()) {
      auto& t = attr.toTensor();
      if (t.requires_grad()) {
        auto detached = t.detach();
        detached.set_requires_grad(false);
        attr = IValue(std::move(detached));
      }
    } else if (attr.isTuple()) {
      auto tuple = std::move(attr).toTuple();
      std::vector<IValue>& elems = tuple->elements();
      for (auto& elem : elems) {
        elem = overrideGradient(elem);
      }
      attr = std::move(tuple);

    } else if (attr.isList()) {
      c10::List<IValue> elems = std::move(attr).toList();
      for (const auto i : c10::irange(elems.size())) {
        elems.set(i, overrideGradient(elems.extract(i)));
      }
      attr = std::move(elems);
    } else if (attr.isGenericDict()) {
      auto dict = std::move(attr).toGenericDict();
      for (const auto& pair : dict) {
        auto val = pair.value();
        val = overrideGradient(val);
      }
      attr = std::move(dict);
    } else if (attr.isObject() && !attr.toObjectRef().type()->is_module()) {
      auto obj_type = attr.type()->expect<ClassType>();
      auto obj_value = std::move(attr).toObject();
      auto sub_attributes = obj_type->getAttributes();
      for (const auto& sub_attr : sub_attributes) {
        auto sub_attr_val = obj_value->getAttr(sub_attr.getName());
        sub_attr_val = overrideGradient(sub_attr_val);
      }
      return obj_value;
    }

    return attr;
  }

  // This method is invoked only when 'freezeInterfaces' parameter is on.
  // The module associated with Interface is retrieved and the invoked method
  // is inlined.
public:
  void propagateAttributes(std::shared_ptr<Graph>& graph) {
    std::unordered_map<ModulePtr, std::unordered_map<std::string, Value*>>
        attrValues;
    auto isEval = !module_.hasattr("training") || !module_.is_training();
    GRAPH_DEBUG("Freezing Module: ", module_.type()->name()->name());
    auto block = graph->block();
    std::stack<Block*> blocks({block});

    Node* m = *block->nodes().begin();
    WithInsertPoint guard(m);
    while (!blocks.empty()) {
      Block* block = blocks.top();
      blocks.pop();
      for (auto it = block->nodes().begin(); it != block->nodes().end();) {
        Node* n = *it;
        it++; // advance iterator bc the current node may be destroyed

        for (Block* sub_block : n->blocks()) {
          blocks.push(sub_block);
        }
        if (n->kind() == prim::GetAttr) {
          auto name = n->s(attr::name);
          auto attrModule = module_;
          auto input = n->inputs()[0];
          
          if (!findConstantAttr(input, name, attrModule, graph)) {
            GRAPH_DEBUG(
                input->type()->cast<InterfaceType>() ||
                        input->type()->expectRef<ClassType>().is_module()
                    ? "attribute: " + name + " is mutable."
                    : "");
            continue;
          }
          std::cout<<"attrModule::"<<name<<std::endl;
          TORCH_INTERNAL_ASSERT(attrModule.hasattr(name));
          Value* paramConst = nullptr;
          auto iter = attrValues.find(attrModule._ivalue());
          if (iter != attrValues.end()) {
            auto iter2 = iter->second.find(name);
            if (iter2 != iter->second.end())
              paramConst = iter2->second;
          }
          if (!paramConst) {
            auto attr = attrModule.attr(name);
            if (!isEval || preserveParameters_) {
              auto type = attrModule.type();
              auto slot = *type->findAttributeSlot(name);
              if (type->is_parameter(slot) || type->is_buffer(slot) ||
                  (attr.isObject() &&
                   !attr.toObjectRef().type()->is_module())) {
                continue;
              } else {
                attr = overrideGradient(attr);
              }
              if (!isEval && name == "training") {
                continue;
              }
            } else {
              attr = overrideGradient(attr);
            }
            if (auto attrVal = tryInsertConstant(*graph, attr)) {
              paramConst = *attrVal;
            } else {
              GRAPH_DEBUG(
                  attr.type()->cast<ClassType>() ? "" : "attribute: ",
                  name,
                  " is not materializable.");
              continue;
            }
            std::string fullName("self.");
            for (auto& name : names_) {
              fullName += name + '.';
            }
            fullName += name;
            paramConst->setDebugName(fullName);
            attrValues[attrModule._ivalue()][name] = paramConst;
          }
          GRAPH_UPDATE(
              "Folding GetAttr %",
              n->outputs()[0]->debugName(),
              " with ",
              paramConst->debugName());
          n->outputs().at(0)->replaceAllUsesWith(paramConst);
          n->removeAllInputs();
        } else if (n->kind() == prim::fork) {
          applyToForkSubgraph(
              n,
              graph,
              // NOLINTNEXTLINE(modernize-avoid-bind)
              std::bind(
                  &AttributePropagator::propagateAttributes,
                  *this,
                  std::placeholders::_1));
        }
      }
    }
  }

  void applyToForkSubgraph(
      Node* n,
      std::shared_ptr<Graph>& graph,
      const std::function<void(std::shared_ptr<Graph>&)>& func) {
    TORCH_CHECK(n->kind() == prim::fork);
    auto attrModule = module_;
    auto node = n->inputs()[0]->node();
    // Check if first parameter of fork is a module. This module is used
    // as the base module (similar to 'self' in forward) to resolve GetAttrs.
    //  Otherwise freezing is applied using module_
    if (node->kind() == prim::GetAttr &&
        node->output()->type()->cast<ClassType>()) {
      auto name = node->s(attr::name);
      auto input = node->inputs()[0];
      if (!findConstantAttr(input, name, attrModule, graph)) {
        // Module needs to be preserved.
        return;
      }
      attrModule = attrModule.attr(name).toModule();
      std::swap(module_, attrModule);
    }

    auto subgraph = n->g(attr::Subgraph);
    func(subgraph);
    module_ = attrModule;
  }
  // Contains attributes that can't be folded or user directs to keep them.
  IValue::HashAliasedIValues preservedAttrs_;
  // Tracked immutable types (Scalars) by their attribute names not
  // IValues.
  std::unordered_map<ModulePtr, std::unordered_set<std::string>>
      preservedScalarAttrs_;

  // Contains user specified methods to be preserved in frozen module.
  std::unordered_set<Function*> preservedMethods_;

  // Contains user specified sub module to be preserve in frozen module.
  std::unordered_set<ModulePtr> preservedSubModule_;

  // Track all used attributes ivalues that can be aliased.
  IValue::HashAliasedIValues usedAttrs_;

  // Contains the attribute slots that need to be preserved for each ClassType.
  std::unordered_map<ClassTypePtr, std::unordered_set<size_t>> attrsToKeep_;

  // Contains the sub modules that share the same ClassType.
  std::unordered_map<ClassTypePtr, IValue::HashAliasedIValues>
      SharedTypeSubModules_;

  Module& module_;

  // Allow to freeze modules containing interfaces.
  bool freezeInterfaces_;

  // Preserve module parameters
  bool preserveParameters_;

  // Contains the attributes names (e.g. {"self", "subModule", "a"}
  std::deque<std::string> names_;
}; // class AttributePropagator

}

}

} //jit

#endif
