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

#ifndef TNN_SOURCE_NETWORK_TNNTORCH_TNNTORCH_PARTITIONING_H
#define TNN_SOURCE_NETWORK_TNNTORCH_TNNTORCH_PARTITIONING_H

#include <vector>

#include "tnn/core/macro.h"
#include "tnn/network/torch/segment.h"
#include "tnn/network/torch/shape_inference.h"
#include "torch/csrc/jit/ir/ir.h"

namespace TNN_NS {
namespace partitioning {

typedef std::vector<SegmentedBlock> PartitionedGraph;

PartitionedGraph segment_graph(std::shared_ptr<torch::jit::Graph> g);

std::vector<SegmentedBlock> Partition(torch::jit::Module& mod, std::shared_ptr<torch::jit::Graph>, NetworkConfig& config);

}  // namespace partitioning
}  // namespace TNN_NS

#endif