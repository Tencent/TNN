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

#define NODE_TEST(expr)                                                                 \
    if (!(expr)) {                                                                      \
        char _ss[2000];                                                                 \
        snprintf(_ss, 2000, "\t\tSanity Check failed on expr "#expr" of node [%s]",     \
                this->name().c_str());                                                  \
        ERROR("%s", _ss);                                                               \
        return Status(TNNERR_COMMON_ERROR, _ss);                                        \
    }


    Edge::Edge(Node * _src, Node * _dst, const std::string &_blob) : src(_src), dst(_dst), tensor_name(_blob) {}


    Node::Node(std::shared_ptr<LayerInfo> &layer_info) {
        info = layer_info;
    }

    Node::Node(const std::string &tensor_name) {
        // create placeholder node 
        info = std::make_shared<LayerInfo>();
        info->type = LAYER_PLACEHOLDER;
        info->name = tensor_name;
        info->outputs = {tensor_name};
    }

    Status Node::addOutputEdge(Edge * e) {
        if (e->src != this) {
            return Status(TNNERR_COMMON_ERROR, "invalid output Edge.");
        }
        output_edges.push_back(e);
        return TNN_OK;
    }

    Status Node::addInputEdge(Edge * e) {
        if (e->dst != this) {
            return Status(TNNERR_COMMON_ERROR, "invalid input Edge.");
        }
        input_edges.push_back(e);
        return TNN_OK;
    }

    Status Node::addInput(Edge * e) {
        RETURN_IF_FAIL(addInputEdge(e));
        info->inputs.push_back(e->tensor_name);
        return TNN_OK;
    }

    Status Node::sanityCheck() {
        NODE_TEST(info->inputs.size() == input_edges.size());
        for(size_t i=0;i<info->inputs.size();i++) {
            NODE_TEST(info->inputs[i] == input_edges[i]->tensor_name);
            NODE_TEST(input_edges[i]->src != nullptr);
            NODE_TEST(input_edges[i]->dst == this);
        }

        auto validOutput = [&](const std::string &name) -> bool {
            return std::find(info->outputs.begin(), info->outputs.end(), name) != info->outputs.end();
        };

        for(size_t i=0;i<output_edges.size();i++) {
            NODE_TEST(validOutput(output_edges[i]->tensor_name));
            NODE_TEST(output_edges[i]->src == this);
            NODE_TEST(output_edges[i]->dst != nullptr);
        }

        if (info->type == LAYER_PLACEHOLDER) {
            NODE_TEST(info->inputs.size() == 0);
            NODE_TEST(info->outputs.size() == 1);
        }

        return TNN_OK;
    }

    Status Graph::fromNetStructure(std::vector<std::shared_ptr<LayerInfo> > layers) {
        *this = Graph();
        for (auto layer : layers) {
            auto node = std::make_shared<Node>(layer);
            nodes.push_back(node);
            for (auto out : layer->outputs) {
                if (tensor_2_node.find(out) != tensor_2_node.end()) {
                    return Status(TNNERR_COMMON_ERROR ,"duplicated tensor_name found.");
                }
                tensor_2_node[out] = node;
            }
            for (auto in : layer->inputs) {
                auto n = getNodeOrCreatePlaceHolder(in);
                auto e = std::make_shared<Edge>(n.get(), node.get(), in);
                RETURN_IF_FAIL(n->addOutputEdge(e.get()));
                RETURN_IF_FAIL(node->addInputEdge(e.get()));
                edges.push_back(e);
            }
        }
        return TNN_OK;
    }

    Graph::Graph(std::string proto_str) {
        // TODO impl, parse a subgraph from prototext, 
        // Could be used as pattern for GraphRewriter 
    }

    std::shared_ptr<Node> Graph::getNodeOrCreatePlaceHolder(const std::string &tensor_name) {
        if (tensor_2_node.find(tensor_name) != tensor_2_node.end()) {
            return tensor_2_node[tensor_name];
        }
        auto input = std::make_shared<Node>(tensor_name);
        placeholders.push_back(input);
        RAISE_ON_ERROR(buildNodeTensorIndex(input));
        return input;
    }

    std::shared_ptr<Node> Graph::getNodeByTensorName(const std::string &tensor_name) const {
        if (tensor_2_node.find(tensor_name) != tensor_2_node.end()) {
            return tensor_2_node.at(tensor_name);
        }
        return nullptr;
    }
    std::shared_ptr<Tensor> Graph::getTensorByName(const std::string &tensor_name) const {
        if (tensors.find(tensor_name) != tensors.end()) {
            return tensors.at(tensor_name);
        }
        return nullptr;
    }

    Status Graph::addNode(const std::shared_ptr<Node> &n) {
        RETURN_ON_NEQ(n->sanityCheck(), TNN_OK);
        nodes.push_back(n);
        RETURN_ON_NEQ(buildNodeTensorIndex(n), TNN_OK);
        return TNN_OK;
    }

    Status Graph::renameTensor(const std::string &old_name, const std::string &new_name) {
        return TNN_OK;
    }

    Status Graph::markOutput(const std::string &tensor_name) {
        if (tensors.find(tensor_name) == tensors.end()) {
            return Status(TNNERR_COMMON_ERROR, "specified tensor not found.");
        }
        marked_outputs.insert(tensor_name);
        return TNN_OK;
    }

    const std::vector<std::weak_ptr<const Node>> Graph::allNodes() const {
        return std::vector<std::weak_ptr<const Node>>(nodes.begin(), nodes.end());
    }

    Status Graph::buildNodeTensorIndex(const std::shared_ptr<Node> n) {
        RETURN_ON_NEQ(n->sanityCheck(), TNN_OK);

        for (auto out : n->info->outputs) {
            if (tensor_2_node.find(out) != tensor_2_node.end()) {
                return Status(TNNERR_COMMON_ERROR, "duplicated tensor_name found.");
            }
            tensor_2_node[out] = n;

            if (tensors.find(out) != tensors.end()) {
                return Status(TNNERR_COMMON_ERROR, "duplicated tensors found.");
            }
            tensors[out] =  std::make_shared<Tensor>(out);
        } 
        for (size_t i=0;i<n->input_edges.size();i++) {
            if (tensor_2_node.find(n->info->inputs[i]) == tensor_2_node.end()) {
                return Status(TNNERR_COMMON_ERROR, "input node not found.");
            }
            if (tensors.find(n->info->inputs[i]) == tensors.end()) {
                return Status(TNNERR_COMMON_ERROR, "input tensor not found.");
            }
            tensor_2_edge[n->info->inputs[i]].push_back(n->input_edges[i]);
        }
        return TNN_OK;
    }

    Status Graph::reBuildTensorIndex() {
        tensor_2_node.clear();
        tensor_2_edge.clear();
        tensors.clear();

        for(auto &n : placeholders) {
            RETURN_IF_FAIL(buildNodeTensorIndex(n));
        }

        for(auto &n : nodes) {
            RETURN_IF_FAIL(buildNodeTensorIndex(n));
        }
        return TNN_OK;
    }

    std::vector<const Tensor*> Graph::getTensorsByNames(const std::vector<std::string> &names) const throw(...) {
        std::vector<const Tensor*> res;
        for(auto &name : names) {
            auto tensor = getTensorByName(name);
            if (!tensor) {
                throw std::runtime_error("got unkonwn tensor.");
            }
            res.push_back(tensor.get());
        }
        return res;
    }

    std::vector<const Tensor*> Graph::outputs_() const {
        std::set<std::string> names = marked_outputs;

        for(auto &n : nodes) {
            if (n->output_edges.size() == 0) {
                for(auto &name : n->info->outputs)
                    names.insert(name);
            }
        }
        return getTensorsByNames(std::vector<std::string>(names.begin(), names.end()));
    }

    std::vector<const Tensor*> Graph::inputs_() const {
        std::set<std::string> names;

        for(auto &n : placeholders) {
            RAISE_ON_ERROR(n->sanityCheck());
            if (n->output_edges.size() > 0) {
                names.insert(n->info->outputs[0]);
            }
        }
        return getTensorsByNames(std::vector<std::string>(names.begin(), names.end()));
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
                    // TODO topological sort the inputs and outputs of the two graphs.
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
        os << "\"1 " << tensor_2_node.size() << " 1 4206624772 ,\"\n";
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
            DEBUG("\tupdateVector origin:%s new_name:%s", origin.c_str(), new_name.c_str());
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
                DEBUG("Updating node edges, here is inputs of Node[%s]", e->dst->name().c_str());
                updateVector(e->dst->info->inputs, e->tensor_name, name_prefix + e->tensor_name);
                e->tensor_name = name_prefix + e->tensor_name;
            }
            updateVector(n->info->outputs);
        };

        // the inEdges and outEdge is calculated according to the edges. 
        // so make a copy here before changing the graph.
        auto in_edges = anchor->inEdges();
        auto out_edges = anchor->outEdges();

        // number of input_edegs of new_graph might less than that of the old graph.
        // so we need to remove the duplicated outputs edges if there is.

        for(auto & e : in_edges) {
            Node * old_node = e->dst;
            Node * new_node = replace_map.at(old_node);
            std::remove_if(old_node->input_edges.begin(), old_node->input_edges.end(), [&](Edge * cur){
                return cur->src == e->src;
            });
            e->dst = new_node;
            DEBUG("Adding input[%s] to Node[%s]", e->tensor_name.c_str(), new_node->name().c_str());
            RAISE_ON_ERROR(new_node->addInput(e));
        }

        for(auto & e : out_edges) {
            Node * old_node = e->src;
            Node * new_node = replace_map.at(old_node);
            std::remove_if(old_node->output_edges.begin(), old_node->output_edges.end(), [&](Edge * cur){
                return cur->dst == e->dst;
            });
            e->src = new_node;
            DEBUG("Updating inputs of Node[%s]", e->dst->name().c_str());
            updateVector(e->dst->info->inputs, e->tensor_name, new_node->info->outputs[0]);
            e->tensor_name = new_node->info->outputs[0];
            RAISE_ON_ERROR(new_node->addOutputEdge(e));
        }

        for(auto &n : nodes) {
            addNamePrefix(n.get());
        }

        auto it = g->nodes.begin();
        for(; it != g->nodes.end();) {
            if (std::find(anchor->nodes.begin(), anchor->nodes.end(), *it) != anchor->nodes.end()) {
                for(auto &tensor_name : (*it)->info->outputs) g->tensor_2_node.erase(tensor_name);
                it = g->nodes.erase(it);
            } else {
                it ++;
            }
        }
        // TODO Remove useless Edges.

        g->nodes.insert(g->nodes.end(), nodes.begin(), nodes.end());
    }
}
