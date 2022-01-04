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

#include "tnn/network/torch/segment.h"

namespace TNN_NS {
namespace partitioning {

SegmentedBlock::SegmentedBlock(SegmentedBlockTarget blk_target, std::vector<torch::jit::Node*>& nodes)
    : target_(blk_target), g_(std::make_shared<torch::jit::Graph>()) {
    for (auto& node : nodes) {
        nodes_.push_back(node);
        appendNode(node);
    }
}

void SegmentedBlock::registerOutput(torch::jit::Value* raw_output) {
    outputs_.push_back(raw_output);
    g_->registerOutput(old_to_new_[raw_output]);
}

void SegmentedBlock::eraseInput(size_t i) {
    inputs_.erase(inputs_.begin() + i);
    g_->eraseInput(i);
}

void SegmentedBlock::eraseOutput(size_t i) {
    outputs_.erase(outputs_.begin() + i);
    g_->eraseOutput(i);
}

torch::jit::Value* SegmentedBlock::getOrAddInputForValue(torch::jit::Value* old_value) {
    if (old_to_new_.count(old_value) == 0) {
        auto node = old_value->node();

        if (node->kind() == torch::jit::prim::Constant) {
            auto new_const = g_->createClone(node, {nullptr});
            g_->block()->prependNode(new_const);
            old_to_new_[old_value] = new_const->output();
            return new_const->output();
        }
        auto new_value = g_->block()->addInput();
        // every time when we addInput, we push back the corresponding lowering graph torch::jit::Value to our
        // raw_inputs
        inputs_.push_back(old_value);
        old_to_new_[old_value] = new_value;
        new_value->copyMetadata(old_value);
        return new_value;
    } else {
        return old_to_new_[old_value];
    }
}

torch::jit::Node* SegmentedBlock::cloneNode(torch::jit::Node* node) {
    auto* block = g_->block();
    auto env    = [&](torch::jit::Value* v) { return getOrAddInputForValue(v); };

    // create node for current graph by using the metadata in node and input Values in env
    auto new_node = block->appendNode(g_->createClone(node, env));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
        auto oo         = node->outputs()[i];
        auto no         = new_node->outputs()[i];
        old_to_new_[oo] = no;
    }
    return new_node;
}

void SegmentedBlock::check_raw_nodes() {
    std::unordered_set<torch::jit::Node*> node_set(nodes_.begin(), nodes_.end());
    nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(),
                                [&](torch::jit::Node* node) {
                                    if (node->kind() == at::aten::size) {
                                        for (auto& use : node->output()->uses()) {
                                            if (node_set.count(use.user))
                                                return false;
                                        }
                                        return true;
                                    }
                                    return false;
                                }),
                 nodes_.end());
}

}  // namespace partitioning
}  // namespace TNN_NS