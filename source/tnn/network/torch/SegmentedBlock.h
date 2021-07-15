#pragma once

#include <vector>

#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace partitioning {

struct SegmentedBlock {
 public:
  enum SegmentedBlockTarget {
    kTorch,
    kTNN,
  };

  SegmentedBlock() = default;
  SegmentedBlock(SegmentedBlockTarget blk_target) : target_(blk_target), g_(std::make_shared<torch::jit::Graph>()) {}
  SegmentedBlock(SegmentedBlockTarget blk_target, std::vector<torch::jit::Node*>& nodes);
  SegmentedBlock(SegmentedBlockTarget blk_target, std::shared_ptr<torch::jit::Graph> g) : target_(blk_target), g_(g) {}

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
  // void register_inshape(std::vector<ir::InputRange>& in_shape) {
  //   in_shape_ = in_shape;
  // }
  // const std::vector<ir::InputRange>& in_shape() const {
  //   return in_shape_;
  // }
  void update_target(SegmentedBlockTarget new_target) {
    target_ = new_target;
  }
  enum SegmentedBlockTarget target() {
    return target_;
  }

 private:
  SegmentedBlockTarget target_;
  // std::vector<ir::InputRange> in_shape_;
  std::vector<torch::jit::Value*> inputs_;
  std::vector<torch::jit::Value*> outputs_;
  std::vector<torch::jit::Node*> nodes_;
  std::shared_ptr<torch::jit::Graph> g_;
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> old_to_new_;
};

} // namespace partitioning
} // namespace trtorch