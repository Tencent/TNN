#include "tnn/network/torch/SegmentedBlock.h"

namespace trtorch {
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
    // every time when we addInput, we push back the corresponding lowering graph torch::jit::Value to our raw_inputs
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
  auto env = [&](torch::jit::Value* v) { return getOrAddInputForValue(v); };

  // create node for current graph by using the metadata in node and input Values in env
  auto new_node = block->appendNode(g_->createClone(node, env));
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    old_to_new_[oo] = no;
  }
  return new_node;
}

} // namespace partitioning
} // namespace trtorch