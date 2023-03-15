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

#include "tnn/optimizer/graph_matcher/graph_matcher.h"

#include <vector>
#include <stack>
#include <map>
#include <set>
#include <sstream>

#include "tnn/core/macro.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/logger.h"

namespace TNN_NS {

const char * pattern_node_prefix = "@";

#define TEST(expr)              \
        if (!(expr)) {          \
            DEBUG("\t\tcheck "#expr" failed for node [%s] with pattern [%s]", \
                    node->name().c_str(), probe->name().c_str());   \
            return false;       \
        }

void AnchorGraph::backTrace(int recursion) {
    for(auto it = paired_nodes.begin(); it!= paired_nodes.end(); ) {
        if (it->second.recursion >= recursion) {
            it = paired_nodes.erase(it);
        } else {
            ++it;
        }
    }
}


bool AnchorGraph::matchUp(const Node *node, Node* probe, int recursion, bool silence) {
    if (paired_nodes.find(node) != paired_nodes.end()) {
        if (paired_nodes.at(node).anchor != probe) {
            if (!silence) DEBUG("node[%s] is already paired with another probe[%s].", node->name().c_str(), 
                        paired_nodes.at(node).anchor->name().c_str());
            return false;
        }
        return true;
    }
    for(auto it : paired_nodes) {
        if (it.second.anchor == probe) {
            if (!silence) DEBUG("probe[%s] is already paired with another node[%s].", probe->name().c_str(), it.second.node->name().c_str());
            return false;
        }
    }

    if (probe->info->type == LAYER_PLACEHOLDER) {
        if (!silence) DEBUG("%*srec[%d] node[%s] matched with pattern placeholder [%s]", (recursion%20)*4, "", recursion, node->name().c_str(), probe->name().c_str());
        paired_nodes[node] = NodePair(node, probe, recursion);
        return true;
    }

    TEST(node->info->type == probe->info->type || probe->info->type == LAYER_ANY_TYPE);
    TEST(node->info->inputs.size() == probe->info->inputs.size());
    TEST(node->input_edges.size() == probe->input_edges.size());
    TEST(node->info->outputs.size() == probe->info->outputs.size());

    for(size_t i=0;i<probe->input_edges.size();i++) {
        TEST(matchUp(node->input_edges[i]->src, probe->input_edges[i]->src, recursion+1, silence));
    }

    if (!silence) DEBUG("%*srec[%d] node[%s] matched with pattern[%s]", (recursion%20)*4, "", recursion, node->name().c_str(), probe->name().c_str());
    paired_nodes[node] = NodePair(node, probe, recursion);

    return true;   
}

std::vector<const Node *> AnchorGraph::allStructualMatchedNodes(const Node * pattern_sibling_node) {
    struct Path {
        const Node * n;
        std::stack<LayerType> types;

        Path(const Node *ptr, std::stack<LayerType> _types=std::stack<LayerType>()) : n(ptr), types(_types) {};
    };
    std::queue<Path> start_points;

    std::stack<Path> que;
    que.push(Path(pattern_sibling_node));
    // DFS to find the common node
    while(!que.empty()) {
        Path candidate = que.top();
        que.pop();

        bool found = false;
        for(auto it = paired_nodes.begin(); it != paired_nodes.end();it++) {
            if (it->second.anchor == candidate.n) {
                DEBUG("add Start point:%s", it->first->name().c_str());
                const Node * sibling_node = it->first;
                start_points.push(Path(sibling_node, candidate.types));
                found = true;
                break;
            }
        }
        if (found) continue;

        Path next(candidate.n, candidate.types);
        next.types.push(candidate.n->info->type);
        for(auto &e : candidate.n->input_edges) {
            next.n = e->src;
            que.push(next);
        }
    }

    std::vector<const Node *> res;
    // BFS to find all matched Nodes
    while(!start_points.empty()) {
        Path path = start_points.front(); start_points.pop();

        std::stringstream ss;
        ss << "test start Point:" << path.n->name() << " type path:";
        auto tmp = path.types;
        while(!tmp.empty()) { ss << "[" << tmp.top() << "] "; tmp.pop();}
        DEBUG("%s", ss.str().c_str());

        if (path.types.empty()) {
            res.push_back(path.n);
            continue;
        }

        for(auto &e: path.n->output_edges) {
            // skip already matched nodes.
            if (paired_nodes.find(e->dst) != paired_nodes.end()) {
                continue;
            }
            if (e->dst->info->type == path.types.top() || path.types.top() == LAYER_PLACEHOLDER 
                || path.types.top() == LAYER_ANY_TYPE) 
            {
                std::stack<LayerType> next_types = path.types;
                next_types.pop();
                start_points.push(Path(e->dst, next_types));
            }
        }

    }

    return res;
}

void AnchorGraph::formalize(Graph *g) {
    // copy subgraph related nodes, edges, const_map, layer_resource.
    // Note that :
    //     1.the AnchorGraph do not store matched placeholders, but create fake ones for the inEdges.
    //     2.here we do not modify the original Graph, Nodes and Edges.
    //     3.the AnchorGraph do not use marked_outputs.

    *dynamic_cast<Graph*>(this) = Graph();
    tnn_structure = g->tnn_structure;
    tnn_resource = g->tnn_resource;

    // removed pairs that anchor is a placeholder
    for(auto it = paired_nodes.begin(); it!= paired_nodes.end(); ) {
        if (it->second.anchor->info->type == LAYER_PLACEHOLDER) {
            it = paired_nodes.erase(it);
        } else {
            ++it;
        }
    }

    for(auto &n : g->nodes) {
        if (paired_nodes.find(n.get()) != paired_nodes.end()) {
            nodes.push_back(n);
            for(auto &name : n->info->outputs) tensor_2_node[name] = n;
        }
    }
    for(auto &e : g->edges) {
        if (paired_nodes.find(e->src) != paired_nodes.end() || paired_nodes.find(e->dst) != paired_nodes.end()) {
            edges.push_back(e);
        }
    }
    // placeholder types is not included in the paired nodes now.
    for(auto &n : g->placeholders) {
        if (paired_nodes.find(n.get()) != paired_nodes.end()) {
            throw std::runtime_error("PlaceHolder node should not be included in the AnchorGraph.");
        }
    }

    // Create input Tensors from the edges. multiple edges may point to the same node.
    for(auto &e : inEdges()) {
        if (tensor_map.find(e->tensor_name) == tensor_map.end()) {
            // this function will build the tensor
            getNodeOrCreatePlaceHolder(e->tensor_name);
        }
    }

    RAISE_ON_ERROR(createUnspecifiedTensors());
    RAISE_ON_ERROR(reBuildTensorIndex());

    auto getShared = [&](const Node *p) {
        for(auto &n:nodes) {if (n.get() == p) return n;}
        throw std::runtime_error("Not found the paired nodes in the subgraph.");
    };

    for(auto it = paired_nodes.begin(); it!= paired_nodes.end(); it++) {
        for(auto &name : it->second.anchor->info->outputs) {
            if (!isdigit(name[0])) {
                // TODO :
                //      Change to mark the node_name, rather than tensor_name.
                std::string ref_name = std::string(pattern_node_prefix) + name;
                tensor_2_node[ref_name] = getShared(it->first);
            }
        }
    }
}

Status AnchorGraph::sanityCheck() {
    for(auto &n : placeholders) {
        RETURN_ON_FAIL(n->sanityCheck());
    }

    for(auto &n : nodes) {
        RETURN_ON_FAIL(n->sanityCheck());
    }
    return TNN_OK;
};

Status AnchorGraph::setInputsOrder(std::vector<std::string> tensor_names) {
    std::set<std::string> names_set(tensor_names.begin(), tensor_names.end());
    if (names_set.size() != tensor_names.size()) {
        ERRORV("AnchorGaaph::setInputsOrder got dulicated tensor names", msg);
        return Status(TNNERR_COMMON_ERROR, msg);
    }
    input_order.clear();
    if (names_set.size() != inputs().size()) {
        ERRORV("In AnchorGraph::setInputsOrder, number of tensors not match", msg);
        return Status(TNNERR_COMMON_ERROR, msg);
    }

    auto in_edges = inEdges();
    auto findNode = [&](const std::string name) -> std::pair<const Node *, size_t> {
        // return the first Node that reference this tensor
        for(auto e: in_edges) {
            if (e->tensor_name == name) {
                size_t offset = 0;
                while(offset < e->dst->info->inputs.size()) {
                    if (e->dst->info->inputs[offset] == name)
                        break;
                    offset ++;
                }
                if (offset == e->dst->info->inputs.size()) 
                    return std::make_pair(nullptr, 0);
                return std::make_pair(e->dst, offset);
            }
        }
        return std::make_pair(nullptr, 0);
    };

    for(auto name : tensor_names) {
        auto p = findNode(name);
        if (p.first == nullptr) {
            ERRORV("AnchorGaaph::setInputsOrder got invalid tensor name: %s", msg, name.c_str());
            return Status(TNNERR_COMMON_ERROR, msg);
        }
        input_order.push_back(p);
    }
    return TNN_OK;
};

Status AnchorGraph::getIOOrderingOfPatternGraph(Graph * pattern_graph, std::vector<std::string> &_input_order, std::vector<std::string> &_output_order) {
    auto findPairedNode= [&](const Node * probe_node ) -> const Node * {
        for(auto p : paired_nodes) {
            if (p.second.anchor == probe_node) {
                return p.second.node;
            }
        }
        return nullptr;
    };

    auto findPairedInputTensor = [&](const std::string pattern_tensor_name) -> std::string {
        auto pattern_placeholder = pattern_graph->getNodeByTensorName(pattern_tensor_name);
        if (!pattern_placeholder) {
            ERRORV("pattern input tensor %s not found.", msg, pattern_tensor_name.c_str());
            throw std::runtime_error(msg);
        }
        if (pattern_placeholder->output_edges.size() == 0) {
            ERRORV("pattern tensor %s is not referenced.", msg, pattern_tensor_name.c_str());
            throw std::runtime_error(msg);
        }
        auto pattern_node = pattern_placeholder->output_edges[0]->dst;
        auto paired_node = findPairedNode(pattern_node);
        auto paired_placeholder = findPairedNode(pattern_placeholder.get());
        if (!paired_node || !paired_placeholder ) {
            ERRORV("paired node or paired_placeholder not found on pattern_tensor:%s.", msg, pattern_tensor_name.c_str());
            throw std::runtime_error(msg);
        }
        for(auto e : paired_node->input_edges) {
            if (e->src == paired_placeholder) {
                return e->tensor_name;
            }
        }
        ERRORV("paired node and paired_placeholder not match on pattern_tensor:%s.", msg, pattern_tensor_name.c_str());
        throw std::runtime_error(msg);
    };

    auto findPairedOutputTensor = [&](const std::string pattern_tensor_name) -> std::string {
        auto pattern_node = pattern_graph->getNodeByTensorName(pattern_tensor_name);
        if (!pattern_node) {
            ERRORV("pattern output tensor %s not found.", msg, pattern_tensor_name.c_str());
            throw std::runtime_error(msg);
        }
        auto paired_node = findPairedNode(pattern_node.get());
        if (!paired_node) {
            ERRORV("paired node not found on pattern_tensor:%s.", msg, pattern_tensor_name.c_str());
            throw std::runtime_error(msg);
        }
        size_t offset = 0;
        for(auto name : pattern_node->info->outputs) {
            if (name == pattern_tensor_name) {
                return paired_node->info->outputs[offset];
            }
            offset ++;
        }
        ERRORV("pattern node not found pattern_tensor:%s.", msg, pattern_tensor_name.c_str());
        throw std::runtime_error(msg);
    };


    _input_order.clear();
    _output_order.clear();
    for(auto t : pattern_graph->inputs()) {
        std::string paired_tensor_name = findPairedInputTensor(t->name);
        DEBUG("pattern input %s paired with %s", t->name.c_str(), paired_tensor_name.c_str());
        _input_order.push_back(paired_tensor_name);
    }

    for(auto t : pattern_graph->outputs()) {
        std::string paired_tensor_name = findPairedOutputTensor(t->name);
        DEBUG("pattern output %s paired with %s", t->name.c_str(), paired_tensor_name.c_str());
        _output_order.push_back(paired_tensor_name);
    }

    return TNN_OK;
};

void match(const std::shared_ptr<Graph> graph, const std::shared_ptr<Graph> pattern,  std::vector<std::shared_ptr<AnchorGraph>>  &results)  {
    results.resize(0);

    std::vector<Node *> pattern_outs = pattern->outputNodes();
    for(auto &node_ref : graph->allNodes()) {
        auto n = node_ref.lock();
        if (!n) throw std::runtime_error("invalid node from graph");
        DEBUG("---------------------- test output[0] of pattern on node [%s]", n->name().c_str());

        std::shared_ptr<AnchorGraph> res = std::make_shared<AnchorGraph>();
        if (res->matchUp(n.get(), pattern_outs[0], 0)) {

            struct DFSState {
                int output_id;
                const Node * node;
                DFSState(int id, const Node *n) : node(n), output_id(id) {}
            };

            auto getRecursion = [](int out_id) -> int {return out_id * 100;};

            std::stack<DFSState> que;
            que.push(DFSState(1, n.get()));
            while(!que.empty()) {
                DFSState cur = que.top(); que.pop();
                DEBUG("---------------------- test output[%d] of pattern on node [%s]", cur.output_id, cur.node->name().c_str());
                res->backTrace(getRecursion(cur.output_id - 1));
                if (!res->matchUp(cur.node, pattern_outs[cur.output_id -1], getRecursion(cur.output_id -1), true)) {
                    std::stringstream ss; 
                    ss << "graph matcher found unmatched Node State on" << cur.node->name() << " on output_id:" << cur.output_id;
                    throw std::runtime_error(ss.str());
                }

                if (cur.output_id == pattern_outs.size())  {
                    auto snapshot = std::make_shared<AnchorGraph>(*res);
                    std::vector<std::string> _in_order, _out_order;
                    RAISE_ON_ERROR(snapshot->getIOOrderingOfPatternGraph(pattern.get(), _in_order, _out_order));
                    snapshot->formalize(graph.get());
                    if (snapshot->inputs().size() == pattern->inputs().size()) 
                        RAISE_ON_ERROR(snapshot->setInputsOrder(_in_order));
                    if (snapshot->outputs().size() == pattern->outputs().size()) 
                        RAISE_ON_ERROR(snapshot->setOutputsOrder(_out_order));
                    results.push_back(snapshot);
                    INFO("matched at node [%s] and pattern [%s]", cur.node->name().c_str(), pattern_outs[cur.output_id-1]->name().c_str());
                    continue;
                }

                // Loop over all possible paired output node for the pattern output
                auto possible_outs = res->allStructualMatchedNodes(pattern_outs[cur.output_id]);

                std::stringstream ss; 
                ss << "output[" << cur.output_id << "] candidates:";
                for(auto &n : possible_outs) { ss << "[" << n->name() << "],"; }
                DEBUG("%s", ss.str().c_str());

                for(auto &candidate : possible_outs) {
                    res->backTrace(getRecursion(cur.output_id));
                    if (res->matchUp(candidate, pattern_outs[cur.output_id], getRecursion(cur.output_id))) {
                        que.push(DFSState(cur.output_id +1, candidate));
                    }
                }
            }
        }
    }
    DEBUG("Match finished\n");
}

std::vector<std::vector<std::shared_ptr<AnchorGraph>>> clustering(const std::vector<std::shared_ptr<AnchorGraph>> &matches) {

    std::map<const Node *, int > groups;

    auto allNodeNotSeen = [&](const std::shared_ptr<AnchorGraph> & g) {
        for(auto &n : g->allNodes()) {
            if (groups.find(n.lock().get()) != groups.end()) {
                return false;
            }
        }
        return true;
    };

    auto relatedClusters = [&](const std::shared_ptr<AnchorGraph> & g) -> std::set<int> {
        std::set<int> _c;
        for(auto &n : g->allNodes()) {
            if (groups.find(n.lock().get()) == groups.end()) {
                continue;
            }
            _c.insert(groups.at(n.lock().get()));
        }
        return _c;
    };

    int cnt = 0;
    for(auto &g : matches) {
        if (allNodeNotSeen(g)) {
            // create a cluster 
            int id = cnt ++;
            for(auto &n : g->allNodes()) {
                groups[n.lock().get()] = id;
            }
        } else {
            // merge the related cluster
            auto clusters_to_merge = relatedClusters(g);
            if (clusters_to_merge.size() > 1) {
                int merged_id = *(clusters_to_merge.begin());
                auto it = clusters_to_merge.begin()++;
                for(;it != clusters_to_merge.end(); it++) {
                    for(auto &pair : groups) {
                        if (pair.second == *it) {
                            pair.second = merged_id;
                        }
                    }
                }
            }
        }
    }

    std::map<int, std::vector<std::shared_ptr<AnchorGraph>>> id_to_cluster;
    for(auto &g : matches) {
        int cluster_id = groups[g->allNodes()[0].lock().get()];
        id_to_cluster[cluster_id].push_back(g);
    }

    std::vector<std::vector<std::shared_ptr<AnchorGraph>>> res;
    for(auto it : id_to_cluster) {
        res.push_back(it.second);
    }

    return res;
}

} // namespace tnn