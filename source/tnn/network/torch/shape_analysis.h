#include "tnn/network/torch/SegmentedBlock.h"
#include "torch/csrc/jit/ir/ir.h"

namespace TNN_NS {
namespace partitioning {

std::vector<torch::jit::IValue> generateRandomInputs(InputShapesMap& input_shape);

void runShapeAnalysis(
    std::vector<SegmentedBlock>& segmented_blocks,
    InputShapesMap& input_shape,
    std::shared_ptr<torch::jit::Graph> g);

} // namespace partitioning
} // namespace TNN_NS