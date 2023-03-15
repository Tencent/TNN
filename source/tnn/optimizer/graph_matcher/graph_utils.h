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

#ifndef TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_GRAPH_UTILS_H_
#define TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_GRAPH_UTILS_H_

#include <memory>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <queue>
#include <vector>

#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/optimizer/graph_matcher/ir.h"

namespace TNN_NS {

namespace utils {

template<typename NodeType, typename GraphType>
struct DAGNodeFactory;

template<typename NodeType, typename GraphType>
struct DAGNode {
    const NodeType * storage = nullptr;
    const GraphType * g = nullptr;
    DAGNodeFactory<NodeType, GraphType> * factory = nullptr;

public:
    std::vector<std::shared_ptr<DAGNode>> prevs() const;
    std::vector<std::shared_ptr<DAGNode>> nexts() const;

    DAGNode(const NodeType * _storage, const GraphType * _g, DAGNodeFactory<NodeType, GraphType> * _factory) : storage(_storage), g(_g), factory(_factory) {}
};

template<typename NodeType, typename GraphType>
struct DAGNodeFactory {
    typedef DAGNode<Node, GraphType> AbstractNode;
    std::map<const Node *, std::shared_ptr<AbstractNode>> nodes;

    std::shared_ptr<AbstractNode> getOrCreateNode(const NodeType *ptr, const GraphType * g ) {
        if (nodes.count(ptr) > 0 ) {
            return nodes.at(ptr);
        }
        auto res = std::make_shared<AbstractNode>(ptr, g, this);
        nodes[ptr] = res;
        return res;
    }
};

template<typename NodeType, typename GraphType>
struct UnionFind {
    typedef DAGNode<NodeType, GraphType> AbstractNode;

    std::map<const AbstractNode *,  const AbstractNode *> parrents;
    std::map<const AbstractNode *,  size_t> ranks;
    size_t count = 0;

    DAGNodeFactory<NodeType, GraphType> factory;

public:
    Status Init(const GraphType * g) {
        if (std::is_same<GraphType, AnchorGraph>::value || std::is_same<GraphType, Graph>::value)  {
            const Graph * graph = dynamic_cast<const Graph*>(g);
            if (!graph) {
                throw std::runtime_error("Expected type of TNN::Graph.");
            }
            for(auto it : graph->allNodes()) {
                const NodeType * node = it.lock().get();
                AbstractNode * ptr = factory.getOrCreateNode(node, g).get();
                parrents[ptr] = ptr;
                ranks[ptr] = 0;
            };
            count = graph->allNodes().size();
        } else {
            throw std::runtime_error("UnionFind Only support TNN::Graph.");
        }
        return TNN_OK;
    }

    const AbstractNode * find(const AbstractNode * p) {
        auto parrent = [&](const AbstractNode *p) {
            if (parrents.find(p) == parrents.end()) {
                throw std::runtime_error("UnionFind got unknow Node.");
            }
            return parrents.at(p);
        };

        auto cur = p;
        while(parrent(cur) != cur) {
            cur = parrent(cur);
        }
        parrents[p] = cur;
        return cur;
    }

    void merge(const AbstractNode * a, const AbstractNode * b) {
        auto a_root = find(a);
        auto b_root = find(b);
        if (a_root == b_root) {
            return;
        }

        size_t a_rank = ranks.at(a_root);
        size_t b_rank = ranks.at(b_root);
        if (a_rank == b_rank) {
            parrents[a_root] = b_root;
            ranks[b_root] = ranks[b_root] + 1;
        } else if (a_rank < b_rank)  {
            parrents[a_root] = b_root;
        } else {
            parrents[b_root] = a_root;
        }
        count -= 1;
    }

};

} // union_find

template<typename GraphType>
Status IsConnectedGraph(GraphType * g, bool & result);

Status LeastCommonAncestors(Graph * g, std::vector<Node *> nodes, Node * &result);

Status TopologicalSort(Graph * g);

} // TNN_NS

#endif // TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_GRAPH_UTILS_H_