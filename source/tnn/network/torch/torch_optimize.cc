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

#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>

namespace torch {
namespace jit {
    void RemoveListAppend(Graph* graph, Block* block) {
        for (auto it = block->nodes().begin(); it != block->nodes().end();) {
            auto* node = *it;
            it++;

            for (Block* sub_block : node->blocks()) {
                RemoveListAppend(graph, sub_block);
            }

            if (!(node->kind() == aten::append && node->inputs().at(0)->node()->kind() == prim::ListConstruct)) {
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
        std::vector<Node*> deleted_nodes;

        for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); it++) {
            Node* node = *it;
            for (auto sub_block : node->blocks()) {
                RemoveConcat(sub_block);
            }

            if (node->kind() != at::aten::cat) {
                continue;
            }
            const int concat_input_size = node->inputs()[0]->node()->inputs().size();
            if (concat_input_size > 1) {
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

    void TorchOptPass(std::shared_ptr<Graph>& graph) {
        RemoveListAppend(graph.get(), graph->block());
        RemoveConcat(graph->block());
        RemoveNoneTypeFromTuple(graph->block());
    }
}  // namespace jit
}  // namespace torch