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

#include "tnn/optimizer/graph_matcher/ir.h"

#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/optimizer/graph_matcher/lexer.h"
#include "tnn/optimizer/graph_matcher/graph_matcher.h"
#include "tnn/optimizer/graph_matcher/logger.h"

namespace TNN_NS {

    Edge::Edge(Node * _src, Node * _dst, const std::string &_blob) : src(_src), dst(_dst), tensor_name(_blob) {}


    Node::Node(std::shared_ptr<LayerInfo> &layer_info) {
        info = layer_info;
    }

    Node::Node(const std::string &blob_name) {
        // create placeholder node 
        info = std::make_shared<LayerInfo>();
        info->type = LAYER_PLACEHOLDER;
        info->name = blob_name;
        info->outputs = {blob_name};
    }

    void Node::addOutputEdge(Edge * e) {
        if (e->src != this) {
            throw std::runtime_error("invalid output Edge.");
        }
        output_edges.push_back(e);
    }

    void Node::addInputEdge(Edge * e) {
        if (e->dst != this) {
            throw std::runtime_error("invalid input Edge.");
        }
        input_edges.push_back(e);
    }

    void Node::addInput(Edge * e) {
        addInputEdge(e);
        info->inputs.push_back(e->tensor_name);
    }

    Node * Node::prev(int id) {
        if (id < input_edges.size()) {
            return input_edges[id]->src;
        } else {
            throw std::runtime_error("invalid Node input index.");
        }
    }

    Node * Node::next(int id) {
        if (id < output_edges.size()) {
            return output_edges[id]->dst;
        } else {
            throw std::runtime_error("invalid Node output index.");
        }
    }

    bool Node::matchSequence(std::pair<int, LayerType> * seq, int seq_len, bool reverse) {
        if (seq_len == 0) {
            return true;
        }
        
        int port = seq[0].first;
        LayerType type = seq[0].second;

        std::vector<Edge*> edges = output_edges;
        if (reverse) {
            edges = input_edges;
        }

        if (port == -1) {
            for(auto e: edges) {
                Node * n = e->dst;
                if (reverse) n = e->src;
                if (n->info->type != type) { 
                    continue;
                }
                if (n->matchSequence(seq + 1, seq_len -1, reverse)) {
                    return true;
                }
            }
            return false;
        }

        if (port >= edges.size()) { return false; }
        Node * n = edges[port]->dst;
        if (reverse) {n = edges[port]->src; }

        if (n->info->type != type) {
            return false;
        }
        return edges[port]->src->matchSequence(seq+1, seq_len -1, reverse);
    }

    Graph::Graph(std::vector<std::shared_ptr<LayerInfo> > layers) {
        for (auto layer : layers) {
            auto node = std::make_shared<Node>(layer);
            nodes.push_back(node);
            for (auto out : layer->outputs) {
                if (blob_2_node.find(out) != blob_2_node.end()) {
                    throw std::runtime_error("duplicated tensor_name found.");
                }
                blob_2_node[out] = node;
            }
            for (auto in : layer->inputs) {
                auto n = getNodeByBlobName(in);
                auto e = std::make_shared<Edge>(n.get(), node.get(), in);
                n->addOutputEdge(e.get());
                node->addInputEdge(e.get());
                edges.push_back(e);
            }
        }
    }

    Graph::Graph(std::string proto_str) {
        // TODO impl, parse a subgraph from prototext, 
        // Could be used as pattern for GraphRewriter 
    }

    std::shared_ptr<Node> Graph::getNodeByBlobName(const std::string &blob_name) {
        if (blob_2_node.find(blob_name) != blob_2_node.end()) {
            return blob_2_node[blob_name];
        }
        auto input = std::make_shared<Node>(blob_name);
        placeholders.push_back(input);
        blob_2_node[blob_name] = input;
        return input;
    }
    std::shared_ptr<Node> Graph::peekNodeByBlobName(const std::string &blob_name) const {
        if (blob_2_node.find(blob_name) != blob_2_node.end()) {
            return blob_2_node.at(blob_name);
        }
        return nullptr;
    }

    Status Graph::rewrite(std::shared_ptr<Graph> &pattern, graph_generator generator) {
        try { 
            std::vector<std::shared_ptr<AnchorGraph>> matches;
            match(shared_from_this(), pattern, matches);

            auto groups = clustering(matches);

            INFO("matched subgraphs:%lu groups:%lu", matches.size(), groups.size());

            for(auto &group : groups) {
                if (group.size() != 1) {
                    // currently, only restricted mode is supported. 
                    // which means the pattern should not has multiple isomorphic matches.
                    WARN("a group of %lu matched SubGraphs are ignored because those subgraphs share some common nodes.", group.size());
                    continue;
                }
                auto &origin_graph = group[0];
                auto heir_graph = generator(group[0]);
                if (heir_graph) {
                    if (heir_graph->inputs().size() != origin_graph->inputs().size()) {
                        WARN("Warning: Skiped one replacement. heir_graph and origin graph num inputs not match, %lu != %lu", 
                                    heir_graph->inputs().size(),  origin_graph->inputs().size());
                        continue;
                    }
                    if (heir_graph->outputs().size() != origin_graph->outputs().size()) {
                        WARN("Warning: Skiped one replacement. heir_graph and origin graph num outputs not match, %lu != %lu", 
                                    heir_graph->outputs().size(),  origin_graph->outputs().size());
                        continue;
                    }
                    heir_graph->embed(shared_from_this(), origin_graph, std::string("_rewrited_") + std::to_string(rewrite_count++) + std::string("_"));
                    INFO("replaced an AnchorGraph with HeirGraph");
                } else {
                    WARN("generate heir graph failed");
                }
            }

        } catch (const std::runtime_error& error) {
            ERROR("%s", error.what());
            return Status(TNNERR_COMMON_ERROR, error.what());
        } catch (...) {
            ERROR("Parser got unknow error.");
            return Status(TNNERR_COMMON_ERROR, "Parser got unknow error.");
        }

        return TNN_OK;
    }

    void Graph::dump(std::ostream &os) const {
        // line 1 header line: 1 num_blobs 1 magic_number
        // !!!assume each node has only one output here.
        os << "\"1 " << blob_2_node.size() << " 1 4206624772 ,\"\n";
        // line 2 inputs: ':'.join(name rank dims dtype)
        auto it = placeholders.begin();
        os << "\"" << (*it)->info->outputs[0] << " 0 0 ";
        for(it++;it!=placeholders.end();it++) {
            os << ": " << (*it)->info->outputs[0] << " 0 0 ";
        }
        os << " ,\"\n\"";
        // line 3 blobs
        for(auto &n : placeholders) {
            for (auto &blob : n->info->outputs) 
                os << blob << " ";
        }
        for(auto &n : nodes) {
            for (auto &blob : n->info->outputs) 
                os << blob << " ";
        }
        os << " ,\"\n\"";
        // line 4 outptus
        for(auto &n : nodes) {
            if (n->output_edges.size() == 0) {
                for (auto &blob : n->info->outputs) 
                    os << blob << " ";
            }
        }
        os << " ,\"\n\"";
        // line 5 num layers
        os << nodes.size() << " ,\"\n";
        // layers
        for(auto &n : nodes) {
            os << "\"" << layerTypeName(n->info->type) << " " << n->name();
            os << " " << n->info->inputs.size() << " " << n->info->outputs.size() << " ";
            for(auto &in : n->info->inputs) { os << in << " "; }
            for(auto &out : n->info->outputs) { os << out << " "; }
            os << " ,\"\n";
        }

    }

    HeirGraph::HeirGraph(const AnchorGraph &g): Graph("") {
        throw std::runtime_error("Deep copy constructor is not implemented yet.");
    };

    void HeirGraph::markOutput(Node *ptr) {
        for(auto &n : nodes) {
            if (n.get() == ptr) {
                if (std::find(output_nodes.begin(), output_nodes.end(), ptr) == output_nodes.end()) {
                    output_nodes.push_back(ptr);
                }
                return;
            }
        }
        throw std::runtime_error("got invalid Node ptr int HeirGraph::markOutput");
    }

    void HeirGraph::markReplacement(Node *origin, Node * new_node) {
        if (origin == nullptr || new_node == nullptr) {
            throw std::runtime_error("nullptr is not allowed in HeirGraph::markReplacement");
        }
        if (origin == new_node) {
            throw std::runtime_error("origin node == new_node is not allowed in HeirGraph::markReplacement");
        }
        for(auto &n : nodes) {
            if (n.get() == new_node) {
                replace_map[origin] = new_node;
                return;
            }
        }
        throw std::runtime_error("got invalid Node ptr in HeirGraph::markReplacement");
    }

    void HeirGraph::markAllInOneNode(const AnchorGraph &g) {
        if (nodes.size() != 1) {
            throw std::runtime_error("HeirGraph expected to have exactly one node.");
        }
        Node * n = nodes[0].get();
        for(auto &e : g.inEdges()) {
            markReplacement(e->dst, n);
        }
        for(auto &e : g.outEdges()) {
            markReplacement(e->src, n);
        }
    }


    void HeirGraph::embed(std::shared_ptr<Graph> g, const std::shared_ptr<AnchorGraph> anchor, std::string name_prefix) {
        // check first
        for(auto & e : anchor->inEdges()) {
            if (replace_map.find(e->dst) == replace_map.end()) {
                throw std::runtime_error("Input edge of the subgraph not found replacement.");
            }
        }
        for(auto & e : anchor->outEdges()) {
            if (replace_map.find(e->src) == replace_map.end()) {
                throw std::runtime_error("Output edge of the subgraph not found replacement.");
            }
        }

        auto updateVector = [&](std::vector<std::string> &v, std::string origin = "", std::string new_name="") {
            DEBUG("updateVector origin:%s", origin.c_str());
            for(auto it = v.begin();it!=v.end();it++)  {
                if (*it == origin || origin.length() == 0) {
                    if (new_name.length() == 0) {
                        if (origin.length() == 0) {
                            new_name = name_prefix + *it;
                        } else {
                            new_name = name_prefix + origin;
                        }
                    }
                    DEBUG("\t\tupdateted origin:%s to:%s", origin.c_str(), new_name.c_str());
                    *it = new_name;
                }
            }
        };

        auto addNamePrefix = [&](Node *n) {
            n->info->name = name_prefix + n->info->name;
            for(auto &e : n->output_edges) {
                updateVector(e->dst->info->inputs, e->tensor_name);
            }
            updateVector(n->info->outputs);
        };

        // the inEdges and outEdge is calculated according to the edges. 
        // so make a copy here before changing the graph.
        auto in_edges = anchor->inEdges();
        auto out_edges = anchor->outEdges();

        for(auto & e : in_edges) {
            Node * old_node = e->dst;
            Node * new_node = replace_map.at(old_node);
            std::remove_if(old_node->input_edges.begin(), old_node->input_edges.end(), [&](Edge * cur){
                return cur->src == e->src;
            });
            e->dst = new_node;
            new_node->addInput(e);
        }

        for(auto & e : out_edges) {
            Node * old_node = e->src;
            Node * new_node = replace_map.at(old_node);
            std::remove_if(old_node->output_edges.begin(), old_node->output_edges.end(), [&](Edge * cur){
                return cur->dst == e->dst;
            });
            e->src = new_node;
            updateVector(e->dst->info->inputs, e->tensor_name, new_node->info->outputs[0]);
            e->tensor_name = new_node->info->outputs[0];
            new_node->addOutputEdge(e);
        }

        for(auto &n : nodes) {
            addNamePrefix(n.get());
        }

        auto it = g->nodes.begin();
        for(; it != g->nodes.end();) {
            if (std::find(anchor->nodes.begin(), anchor->nodes.end(), *it) != anchor->nodes.end()) {
                for(auto &blob_name : (*it)->info->outputs) g->blob_2_node.erase(blob_name);
                it = g->nodes.erase(it);
            } else {
                it ++;
            }
        }

        g->nodes.insert(g->nodes.end(), nodes.begin(), nodes.end());
    }
}
