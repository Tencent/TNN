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

#ifndef TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_GRAPH_REGISTRY_H_
#define TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_GRAPH_REGISTRY_H_

#include <vector>
#include <memory>
#include <list>
#include <string>

#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/optimizer/graph_matcher/ir.h"

namespace TNN_NS {

struct Tokenizer;

struct GraphRegistry {

    std::list<Tokenizer*> consumers;
    std::map<std::string, std::shared_ptr<const Graph>> graphs;

    Status registerGraph(const std::string name, std::shared_ptr<const Graph> graph);

    std::shared_ptr<const Graph>  queryGraphByName(const std::string name);

    Status registerTokenizer(Tokenizer * ptr);
    Status unRegisterTokenizer(Tokenizer * ptr);
};

} // namespace TNN_NS

#endif // TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_GRAPH_REGISTRY_H_
