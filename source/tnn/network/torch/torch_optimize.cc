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

#include "torch_optimize.h"
#include "attribute_propagator.h"
#include "constant_propagation.h"
#include "check_qat_mode.h"

#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include "torch/csrc/jit/passes/inliner.h"

#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/create_functional_graphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/fuse_linear.h"
#include "torch/csrc/jit/passes/guard_elimination.h"
#include "torch/csrc/jit/passes/loop_unrolling.h"
#include "torch/csrc/jit/passes/peephole.h"


namespace torch {
namespace jit {
    int GetMaxBlockSize(Block* block) {
        int max_block_size = 0;
        for (auto it = block->nodes().begin(); it != block->nodes().end();) {
            auto* node = *it;
            it++;
            for (Block* sub_block : node->blocks()) {
                max_block_size = std::max(GetMaxBlockSize(sub_block) + 1, max_block_size);
            }
        }

        return max_block_size;
    }

    void RemoveListAppend(Graph* graph, Block* block) {
        auto check_node = [](torch::jit::Node* node) -> bool {
            if (!(node->kind() == aten::append && node->inputs().at(0)->node()->kind() == prim::ListConstruct)) {
                return false;
            }
            if (node->owningBlock() != node->inputs().at(0)->node()->owningBlock()) {
                return false;
            }

            return true;
        };

        int max_block_size = GetMaxBlockSize(block);

        if (max_block_size > 1) {
            return;
        }

        for (auto it = block->nodes().begin(); it != block->nodes().end();) {
            auto* node = *it;
            it++;

            for (Block* sub_block : node->blocks()) {
                RemoveListAppend(graph, sub_block);
            }

            if (!check_node(node)) {
                continue;
            }

            Value* mutated_value = node->inputs().at(0);
            Node* list_node      = mutated_value->node();
            Node* new_list_node  = graph->create(prim::ListConstruct, 1);
            for (Value* input : list_node->inputs()) {
                new_list_node->addInput(input);
            }
            new_list_node->addInput(node->inputs().at(1));
            new_list_node->copyMetadata(list_node);
            new_list_node->insertAfter(node);
            new_list_node->output()->setType(list_node->output()->type());
            mutated_value->replaceAllUsesAfterNodeWith(node, new_list_node->output());
            node->destroy();
        }
    }

    void RemoveConcat(Block* block) {
        auto check_node = [](torch::jit::Node* node) -> bool {
            if (node->kind() != at::aten::cat) {
                return false;
            }
            if (node->inputs()[0]->node()->inputs().size() != 1) {
                return false;
            }

            return true;
        };

        std::vector<Node*> deleted_nodes;

        for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); it++) {
            Node* node = *it;
            for (auto sub_block : node->blocks()) {
                RemoveConcat(sub_block);
            }

            if (!check_node(node)) {
                continue;
            }

            Value* input_value  = node->inputs()[0]->node()->inputs()[0];
            Value* output_value = node->outputs()[0];
            output_value->replaceAllUsesWith(input_value);
            deleted_nodes.push_back(node);
        }

        for (auto del_node : deleted_nodes) {
            del_node->destroy();
        }
    }

    void RemoveNoneTypeFromTuple(Block* block) {
        for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); it++) {
            std::vector<size_t> deleted_inputs_index;
            Node* node = *it;
            for (auto sub_block : node->blocks()) {
                RemoveNoneTypeFromTuple(sub_block);
            }
            if (node->kind() != at::prim::TupleConstruct) {
                continue;
            }
            const int inputs_size = node->inputs().size();
            for (int i = 0; i < inputs_size; i++) {
                if (node->inputs()[i]->type()->kind() == c10::TypeKind::NoneType) {
                    deleted_inputs_index.push_back(i);
                }
            }
            for (const auto& index : deleted_inputs_index) {
                node->removeInput(index);
            }
        }
    }

    void RemoveException(torch::jit::Block* block) {
        auto check_node = [](torch::jit::Node* n) -> bool {
            if (n->blocks().size() != 2) {
                return false;
            }
            auto block0 = n->blocks()[0];
            auto block1 = n->blocks()[1];
            if (block0->outputs().size() != 0 || block1->outputs().size() != 0) {
                // Make sure that the node doesn't actually produce any Value that are
                // used by other nodes
                return false;
            }

            auto block0_start = block0->nodes().begin();
            auto block1_start = block1->nodes().begin();

            // Make sure that there is at least one empty block
            if (block0_start->kind() != prim::Return && block1_start->kind() != prim::Return) {
                return false;
            }

            if ((*block1_start)->kind() == prim::Return) {
                if ((*block0_start)->kind() == prim::RaiseException) {
                    if ((*(++block0_start))->kind() == prim::Return) {
                        // Make sure that block0 is solely just the exception and the return
                        return true;
                    }
                } else if ((*block0_start)->kind() == aten::format &&
                           (*(++block0_start))->kind() == prim::RaiseException) {
                    if ((*(++block0_start))->kind() == prim::Return) {
                        // Make sure that block0 is solely just the exception and the return
                        return true;
                    }
                }
            }

            if ((*block0_start)->kind() == prim::Return) {
                if ((*block1_start)->kind() == prim::RaiseException) {
                    if ((*(++block1_start))->kind() == prim::Return) {
                        // Make sure that block0 is solely just the exception and the return
                        return true;
                    }
                } else if ((*block1_start)->kind() == aten::format &&
                           (*(++block1_start))->kind() == prim::RaiseException) {
                    if ((*(++block1_start))->kind() == prim::Return) {
                        // Make sure that block0 is solely just the exception and the return
                        return true;
                    }
                }
            }

            return false;
        };

        for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end; ++it) {
            for (auto b : it->blocks()) {
                RemoveException(b);
            }

            if (it->kind() == prim::If && check_node(*it)) {
                it.destroyCurrent();
            }
        }
    }

    void RemoveSlice(Block* block) {
        std::vector<Node*> deleted_nodes;

        for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); it++) {
            Node* node = *it;
            for (auto sub_block : node->blocks()) {
                RemoveSlice(sub_block);
            }

            if (node->kind() != at::aten::slice) {
                continue;
            }

            const auto& inputs = node->inputs();
            const auto dim     = TNN_NS::conversion::getValue<int64_t>(inputs[1]);
            const auto start   = TNN_NS::conversion::getValue<int64_t>(inputs[2]);
            const auto end     = TNN_NS::conversion::getValue<int64_t>(inputs[3]);
            const auto step    = TNN_NS::conversion::getValue<int64_t>(inputs[4]);
            if (dim != 0 || start != 0 || step != 1 || end != LONG_LONG_MAX) {
                continue;
            }

            Value* input_value  = node->inputs()[0];
            Value* output_value = node->outputs()[0];
            output_value->replaceAllUsesWith(input_value);
            deleted_nodes.push_back(node);
        }

        for (auto del_node : deleted_nodes) {
            del_node->destroy();
        }
    }

    void RemoveClone(Block* block) {
        std::vector<Node*> deleted_nodes;

        for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); it++) {
            Node* node = *it;
            for (auto block : node->blocks()) {
                RemoveClone(block);
            }
            if ((node->kind() == c10::Symbol::fromQualString("aten::clone"))) {
                Value* input_value = node->inputs()[0];
                Value* output_value = node->outputs()[0];
                output_value->replaceAllUsesWith(input_value);
                deleted_nodes.push_back(node);
            }
        }
        for (auto del_node : deleted_nodes) {
            del_node->destroy();
        }
    }

    void RemoveContiguous(std::shared_ptr<Graph> graph) {
        std::string contiguous_pattern    = R"IR(
        graph(%input, %1):
            %2 = aten::contiguous(%input, %1)
            return (%2))IR";
        std::string no_contiguous_pattern = R"IR(
        graph(%input, %1):
            return (%input))IR";

        // remove contiguous
        torch::jit::SubgraphRewriter remove_contiguous;
        remove_contiguous.RegisterRewritePattern(contiguous_pattern, no_contiguous_pattern);
        remove_contiguous.runOnGraph(graph);
    }

    void TorchOptPass(script::Module& module) {

        module.eval();
        auto graph = module.get_method("forward").graph();
        if (CheckQatMode(*graph)) {
            // QAT cannot use freeze_module, because freeze_module will remove fake_quantize op
            torch::jit::Inline(*graph);
            ConstantPropagationImmutableTypes(graph);
            std::cout<<"Graph after ConstantPropagation"<<std::endl;
            std::cout << graph->toString(false) << std::endl;

            AttributePropagator propagator(module);
            propagator.propagateAttributes(graph);
            std::cout<<"Graph after AttributePropagator"<<std::endl;
            std::cout << graph->toString(false) << std::endl;
        } else {
            module = torch::jit::freeze_module(module);
            std::cout << graph->toString(false) << std::endl;
        }

        /*
        torch::jit::EliminateRedundantGuards(graph);
        torch::jit::RemoveListMutation(graph);
        torch::jit::RemoveTensorMutation(graph);
        torch::jit::CreateFunctionalGraphs(graph);
        torch::jit::InlineFunctionalGraphs(graph);
        torch::jit::PeepholeOptimize(graph, false);
        torch::jit::FuseLinear(graph);
        torch::jit::LowerAllTuples(graph);
        torch::jit::EliminateDeadCode(graph);
        */ 
        
        LowerSimpleTuples(graph);
        
        removeDropout(module);
        RemoveException(graph->block());
        RemoveListAppend(graph.get(), graph->block());
        RemoveConcat(graph->block());
        RemoveContiguous(graph);
        
//        RemoveClone(graph->block());
//        RemoveNoneTypeFromTuple(graph->block());
//        RemoveSlice(graph->block());

        torch::jit::EliminateDeadCode(graph);
    }
}  // namespace jit
}  // namespace torch
