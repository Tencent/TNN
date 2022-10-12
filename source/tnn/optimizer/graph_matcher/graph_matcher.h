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

#ifndef TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_GRAPH_MATCHER_H_
#define TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_GRAPH_MATCHER_H_

#include <memory>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <queue>
#include <vector>

#include "tnn/core/macro.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/logger.h"

namespace TNN_NS {

extern const char * pattern_node_prefix;

struct NodePair {
    const Node * node = nullptr;
    Node * anchor = nullptr;
    int recursion = 0;

    NodePair(const Node *n, Node *a, int recur) : node(n), anchor(a), recursion(recur) {};
    NodePair(const NodePair &) = default;
    NodePair() = default;
};

struct AnchorGraph : public Graph {
    std::map<const Node *, NodePair> paired_nodes;
    std::vector<std::pair<const Node *, size_t>> input_order;

    AnchorGraph(): Graph() {};
    AnchorGraph(const AnchorGraph &g): Graph(g), paired_nodes(g.paired_nodes) {};
    AnchorGraph& operator=(const AnchorGraph &g)=delete;

    virtual Status sanityCheck() override;

    bool matchUp(const Node *node, Node* probe, int recursion, bool silence=false);

    void backTrace(int recursion);

    std::vector<const Node *> allStructualMatchedNodes(const Node * pattern_sibling_node);

    Status getIOOrderingOfPatternGraph(Graph *g, std::vector<std::string> & in, std::vector<std::string> & out);
    void formalize(Graph *g);

    bool inSubGraph(Edge * e) const {
        bool src_inside = paired_nodes.find(e->src) != paired_nodes.end();
        bool dst_inside = paired_nodes.find(e->dst) != paired_nodes.end();
        return src_inside && dst_inside;
    }

    std::vector<Edge *> outEdges() const {
        std::vector<Edge *> res;
        for(auto &e : edges) {
            if (paired_nodes.find(e->dst) == paired_nodes.end()) {
                res.push_back(e.get());
            }
        }
        return res;
    }

    std::vector<Edge *> inEdges() const {
        std::vector<Edge *> res;
        for(auto &e : edges) {
            if (paired_nodes.find(e->src) == paired_nodes.end()) {
                res.push_back(e.get());
            }
        }
        return res;
    }

    virtual std::vector<Node *> outputNodes() const override {
        // all nodes that have no out edges.
        std::vector<Node *> res = Graph::outputNodes();
        // add those nodes that has an outer edge
        for(auto &e : outEdges()) {
            if (std::find(res.begin(), res.end(), e->src) == res.end()) {
                res.push_back(e->src);
            }
        }
        return res;
    }

    virtual std::vector<Node *> inputNodes() const override {
        std::vector<Node *> res;
        for(auto &e : inEdges()) {
            if (std::find(res.begin(), res.end(), e->src) == res.end()) {
                res.push_back(e->src);
            }
        }
        return res;
    }

    virtual std::vector<const Tensor*> outputs() const override {
        // all tensors that is not used + outEdges
        std::set<std::string> names;

        for(auto &e : outEdges()) names.insert(e->tensor_name);
        for(auto pair : tensor_map) {
            if (tensor_2_edge.count(pair.first) == 0) {
                names.insert(pair.first);
            }
        }

        if (output_order.size() == 0) {
            return getTensorsByNames(std::vector<std::string>(names.begin(), names.end()));
        }
        RAISE_ON_ERROR(validateSetAndVector(names, output_order));
        return getTensorsByNames(output_order);
    }

    virtual std::vector<const Tensor*> inputs() const override {
        // all inEdges
        std::set<std::string> names;

        for(auto &e : inEdges()) names.insert(e->tensor_name);

        if (input_order.size() == 0) {
            return getTensorsByNames(std::vector<std::string>(names.begin(), names.end()));
        }

        std::vector<std::string> ordered_names;
        for(auto p : input_order) ordered_names.push_back(p.first->info->inputs[p.second]);
        RAISE_ON_ERROR(validateSetAndVector(names, ordered_names));
        return getTensorsByNames(ordered_names);
    }

    virtual Status setInputsOrder(std::vector<std::string> tensor_names) override;

};

void match(const std::shared_ptr<Graph> graph, const std::shared_ptr<Graph> pattern, std::vector<std::shared_ptr<AnchorGraph>> &results) ;

std::vector<std::vector<std::shared_ptr<AnchorGraph>>> clustering(const std::vector<std::shared_ptr<AnchorGraph>> &matches);

}

#endif // TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_GRAPH_MATCHER_H_