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

#ifndef TNN_SOURCE_NETWORK_TNNTORCH_CONSTANT_PROPAGATION_H
#define TNN_SOURCE_NETWORK_TNNTORCH_CONSTANT_PROPAGATION_H

#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Runs constant propagation on all objects unless ignore_custom_classes is
// specified as true, in which case user defined classes are skipped.  This is
// useful to prevent early fusion of packing operations, which end up lowering
// away information about their constructors (e.g. packed::linear_clamp_prepack
// and prepacked::conv2d_clamp_prepack)
// Returns True if the pass made a change to the graph
bool ConstantPropagation(
    std::shared_ptr<Graph>& graph,
    bool ignore_custom_classes = false);

// runs constant propagation only on ops that have non-aliasing inputs & outputs
// Returns True if the pass made a change to the graph
bool ConstantPropagationImmutableTypes(std::shared_ptr<Graph>& graph);

// Runs the node if its inputs are constants. Callers of this function must
// make their own determination if constant prop is appropriate - for example
// non-deterministic ops or ops with side effects.  If ignore_custom_classes is
// specified, nodes that output user defined classes are not run.
c10::optional<Stack> runNodeIfInputsAreConstant(
    const Node* node,
    bool ignore_custom_classes = false);

} // namespace jit
} // namespace torch

#endif
