#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Inline function and method calls.
bool CheckQatMode(Graph& graph);

} // namespace jit
} // namespace torch
