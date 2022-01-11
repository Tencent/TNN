#ifndef TNN_SOURCE_TNN_NETWORK_TNN_OPTIMIZE_H
#define TNN_SOURCE_TNN_NETWORK_TNN_OPTIMIZE_H

#include <torch/script.h>

#include "tnn/network/torch/torch_op_converter.h"

namespace torch {
namespace jit {
    void TNNOptPass();
}
}  // namespace torch

#endif  // TNN_SOURCE_TNN_NETWORK_TNN_OPTIMIZE_H
