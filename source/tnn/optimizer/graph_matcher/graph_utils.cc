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

#include "tnn/optimizer/graph_matcher/graph_utils.h"

#include <vector>
#include <map>
#include <list>
#include <queue>

#include "tnn/core/macro.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/graph_matcher.h"
#include "tnn/optimizer/graph_matcher/logger.h"

namespace TNN_NS {

namespace utils {


template<typename GraphType>
struct DAGNode<Node, GraphType> {
    const Node* storage = nullptr;
    const GraphType * g = nullptr;
    DAGNodeFactory<Node, GraphType> * factory = nullptr;

public:
    std::vector<std::shared_ptr<DAGNode>> prevs() const;
    std::vector<std::shared_ptr<DAGNode>> nexts() const;

    DAGNode(const Node* _storage, const GraphType * _g, DAGNodeFactory<Node, GraphType> * _factory) : storage(_storage), g(_g), factory(_factory) {}
};


template<typename GraphType>
std::vector<std::shared_ptr<DAGNode<Node,GraphType>>> DAGNode<Node,GraphType>::prevs() const {

    std::vector<std::shared_ptr<DAGNode<Node,GraphType>>> res;

    for(auto e : storage->input_edges) {
        if (std::is_same<GraphType, AnchorGraph>::value)  {
            const AnchorGraph * anchor_graph = dynamic_cast<const AnchorGraph*>(g);
            if (!anchor_graph) {
                throw std::runtime_error("Expected an AnchorGraph.");
            }
            if (!anchor_graph->inSubGraph(e)) {
                continue;
            }
        }

        auto dag_node = factory->getOrCreateNode(e->src, g);
        res.push_back(dag_node);
    }
    return res;
}

template<typename GraphType>
std::vector<std::shared_ptr<DAGNode<Node,GraphType>>> DAGNode<Node,GraphType>::nexts() const {

    std::vector<std::shared_ptr<DAGNode<Node,GraphType>>> res;

    for(auto e : storage->output_edges) {
        if (std::is_same<GraphType, AnchorGraph>::value)  {
            const AnchorGraph * anchor_graph = dynamic_cast<const AnchorGraph*>(g);
            if (!anchor_graph) {
                throw std::runtime_error("Expected an AnchorGraph.");
            }
            if (!anchor_graph->inSubGraph(e)) {
                continue;
            }
        }

        auto dag_node = factory->getOrCreateNode(e->dst, g);
        res.push_back(dag_node);
    }
    return res;
}

} // namespace utils

template<typename GraphType>
Status IsConnectedGraph(GraphType * g, bool & result) {
    utils::UnionFind<Node, GraphType> uf;
    uf.Init(g);

    std::map<const Node * , bool> visited;
    auto is_visited = [&](const Node * ptr) { return visited.count(ptr) > 0; };
    
    for(auto n : g->allNodes()) {
        auto ptr = n.lock().get();
        if (is_visited(ptr)) {
            continue; 
        }

        // DFS
        std::stack<std::pair<const Node *, const Node *>> que;
        que.push(std::make_pair(ptr, ptr));

        while(!que.empty()) {
            const Node * cur_ptr = que.top().first;
            const Node * root_ptr = que.top().second;
            que.pop();

            // merge two parts
            auto cur = uf.factory.getOrCreateNode(cur_ptr, g);
            auto root = uf.factory.getOrCreateNode(root_ptr, g);
            uf.merge(cur.get(), root.get());

            if (!is_visited(cur_ptr)) {
                for(auto next : cur->nexts()) {
                    que.push(std::make_pair(next->storage, root_ptr));
                }
            }
            visited[cur_ptr] = true;
        }
    }

    result = (uf.count == 1);

    return TNN_OK;
}

template Status IsConnectedGraph(Graph* g, bool & result);
template Status IsConnectedGraph(AnchorGraph* g, bool & result);

} // namespace tnn