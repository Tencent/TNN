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

#ifndef TNN_SOURCE_NETWORK_TNNTORCH_TNNTORCH_SEGMENTBLOCK_H
#define TNN_SOURCE_NETWORK_TNNTORCH_TNNTORCH_SEGMENTBLOCK_H

#include <vector>

#include "tnn/core/blob.h"
#include "tnn/core/macro.h"
#include "torch/csrc/jit/ir/ir.h"

namespace TNN_NS {
namespace partitioning {

struct SegmentedBlock {
public:
    enum SegmentedBlockTarget {
        kTorch,
        kTNN,
    };

    SegmentedBlock() = default;
    SegmentedBlock(SegmentedBlockTarget blk_target)
        : target_(blk_target), g_(std::make_shared<torch::jit::Graph>()) {}
    SegmentedBlock(SegmentedBlockTarget blk_target, std::vector<torch::jit::Node*>& nodes);
    SegmentedBlock(SegmentedBlockTarget blk_target, std::shared_ptr<torch::jit::Graph> g)
        : target_(blk_target), g_(g) {}

    torch::jit::Value* getOrAddInputForValue(torch::jit::Value* v);
    torch::jit::Node* cloneNode(torch::jit::Node* node);
    void appendNode(torch::jit::Node* n) {
        cloneNode(n);
    }
    void registerOutput(torch::jit::Value* raw_output);
    torch::jit::graph_node_list nodes() {
        return g_->nodes();
    }
    const std::vector<torch::jit::Node*>& raw_nodes() const {
        return nodes_;
    }
    torch::jit::Block* block() {
        return g_->block();
    }
    std::shared_ptr<torch::jit::Graph>& g() {
        return g_;
    }
    void update_graph(std::shared_ptr<torch::jit::Graph> new_g) {
        g_ = new_g;
    }
    c10::ArrayRef<torch::jit::Value*> inputs() {
        return g_->inputs();
    }
    c10::ArrayRef<torch::jit::Value*> outputs() {
        return g_->outputs();
    }
    const std::vector<torch::jit::Value*>& raw_inputs() const {
        return inputs_;
    }
    const std::vector<torch::jit::Value*>& raw_outputs() const {
        return outputs_;
    }
    void eraseInput(size_t i);
    void eraseOutput(size_t i);
    bool contain_raw_value(torch::jit::Value* input) {
        return old_to_new_.count(input);
    }
    void register_min_inshape(std::vector<DimsVector>& in_shape) {
        min_in_shape_ = in_shape;
    }
    void register_max_inshape(std::vector<DimsVector>& in_shape) {
        max_in_shape_ = in_shape;
    }
    void register_intype(std::vector<DataType>& in_type) {
        in_type_ = in_type;
    }
    const std::vector<DimsVector>& min_in_shape() const {
        return min_in_shape_;
    }
    const std::vector<DimsVector>& max_in_shape() const {
        return max_in_shape_;
    }
    const std::vector<DataType>& in_type() const {
        return in_type_;
    }
    void update_target(SegmentedBlockTarget new_target) {
        target_ = new_target;
    }
    enum SegmentedBlockTarget target() {
        return target_;
    }

private:
    SegmentedBlockTarget target_;
    std::vector<DimsVector> min_in_shape_;
    std::vector<DimsVector> max_in_shape_;
    std::vector<DataType> in_type_;
    std::vector<torch::jit::Value*> inputs_;
    std::vector<torch::jit::Value*> outputs_;
    std::vector<torch::jit::Node*> nodes_;
    std::shared_ptr<torch::jit::Graph> g_;
    std::unordered_map<torch::jit::Value*, torch::jit::Value*> old_to_new_;
};

}  // namespace partitioning
}  // namespace TNN_NS

#endif