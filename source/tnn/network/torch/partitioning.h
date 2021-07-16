#pragma once

#include <vector>
#include "tnn/core/macro.h"

#include "tnn/network/torch/SegmentedBlock.h"
#include "torch/csrc/jit/ir/ir.h"

namespace TNN_NS {
namespace partitioning {

typedef std::vector<SegmentedBlock> PartitionedGraph;

PartitionedGraph segment_graph(std::shared_ptr<torch::jit::Graph> g);

std::vector<SegmentedBlock> Partition(std::shared_ptr<torch::jit::Graph>, InputShapesMap& input_shape);

} // namespace partitioning
} // namespace TNN_NS