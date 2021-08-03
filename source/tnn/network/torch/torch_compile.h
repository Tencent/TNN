#pragma once

#include "tnn/network/torch/partitioning.h"

namespace TNN_NS {

std::shared_ptr<torch::jit::Module> CompileTorch(std::shared_ptr<torch::jit::Module> mod, InputShapesMap& input_shape,
                                                 NetworkConfig& config);

}