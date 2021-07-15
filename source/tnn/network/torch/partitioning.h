#pragma once

#include <vector>

#include "tnn/network/torch/SegmentedBlock.h"
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace partitioning {

typedef std::vector<SegmentedBlock> PartitionedGraph;

PartitionedGraph segment_graph(std::shared_ptr<torch::jit::Graph> g);

std::vector<SegmentedBlock> Partition(std::shared_ptr<torch::jit::Graph>);

} // namespace partitioning
} // namespace trtorch