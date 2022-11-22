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

#include "tnn/optimizer/net_optimizer_context_marker.h"

#include <map>
#include <memory>
#include <tuple>
#include <set>
#include <vector>

#include "tnn/core/common.h"
#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"

namespace TNN_NS {

namespace optimizer {

namespace context_marker {

    struct Node;

    struct Edge {
        Edge(Node * _src, Node * _dst) : src(_src), dst(_dst) {}
    public:
        Node * src;
        Node * dst;
    };


    struct Node {
        Node(std::shared_ptr<LayerInfo> &layer_info) {
            info = layer_info;
            name = info->name;
        }

        Node(std::string &blob_name) {
            info = std::make_shared<LayerInfo>();
            info->type = LAYER_NOT_SUPPORT;
            name = blob_name;
        }

        void addOutputEdge(Edge * e) {
            output_edges.push_back(e);
        }

        void addInputEdge(Edge * e) {
            input_edges.push_back(e);
        }

        Node * prev(int id) {
            if (id < input_edges.size()) {
                return input_edges[id]->src;
            } else {
                throw std::runtime_error("invalid Node input index.");
            }
        }

        Node * next(int id) {
            if (id < output_edges.size()) {
                return output_edges[id]->dst;
            } else {
                throw std::runtime_error("invalid Node output index.");
            }
        }

        bool matchSequence(std::pair<int, LayerType> * seq, int seq_len, bool reverse) {
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
                        // printf("\t\t\tseq_len:%d skip:<%s> on %d != %d\n",seq_len, n->info->name.c_str(), n->info->type, type);
                        continue;
                    }
                    if (n->matchSequence(seq + 1, seq_len -1, reverse)) {
                        return true;
                    }
                }
                // printf("\t\tseq_len:%d for loop failed at:<%s>\n", seq_len, info->name.c_str());
                return false;
            }

            if (port >= edges.size()) { return false; }
            Node * n = edges[port]->dst;
            if (reverse) {n = edges[port]->src; }

            if (n->info->type != type) {
                // printf("\t\tseq_len:%d type not match at:<%s> %d != %d\n", seq_len, n->info->name.c_str(), n->info->type, type);
                return false;
            }
            return edges[port]->src->matchSequence(seq+1, seq_len -1, reverse);
        }

    public:
        std::string name;
        std::shared_ptr<LayerInfo> info;
        std::vector<Edge*> output_edges;
        std::vector<Edge*> input_edges;
    };

    struct Graph {

        Graph(std::vector<std::shared_ptr<LayerInfo>> layers) {
            for (auto layer : layers) {
                // printf("Construct Node <%s>\n", layer->name.c_str());
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
                    auto e = std::make_shared<Edge>(n.get(), node.get());
                    n->addOutputEdge(e.get());
                    node->addInputEdge(e.get());
                    edges.push_back(e);
                }
            }
        }

        Graph(std::string proto_str) {
            // TODO impl, parse a subgraph from prototext, 
            // Could be used as pattern for GraphRewriter 
        }

        std::shared_ptr<Node> getNodeByBlobName(std::string &blob_name) {
            if (blob_2_node.find(blob_name) != blob_2_node.end()) {
                return blob_2_node[blob_name];
            }
            auto input = std::make_shared<Node>(blob_name);
            placeholders.push_back(input);
            blob_2_node[blob_name] = input;
            return input;
        }

        typedef std::function<std::shared_ptr<Graph>(std::shared_ptr<Graph>)> graph_generator;

        void Rewrite(std::shared_ptr<Graph> &pattern, graph_generator generator) {
            // TODO Impl 
        }


    public:
        std::vector<std::shared_ptr<Node>> nodes;
        std::vector<std::shared_ptr<Edge>> edges;
        std::vector<std::shared_ptr<Node>> placeholders;
        std::map<std::string, std::shared_ptr<Node> > blob_2_node;
    };

    NetOptimizerRegister<NetOptimizerContextMarker> g_net_optimizer_bert_ffn_marker(OptPriority::P1);

    std::string NetOptimizerContextMarker::Strategy() {
        return kNetOptimizerContextMarker;
    }

    bool NetOptimizerContextMarker::IsSupported(const NetworkConfig &net_config) {
        return true;
    }

    Status NetOptimizerContextMarker::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        try {
            auto graph = std::make_shared<Graph>(structure->layers);

            // bool(reverse) indicates this squence is backward or not
            typedef std::tuple<std::vector<std::pair<int, LayerType>>, bool, std::string> NodeSequence;

            std::map<LayerType, std::vector<NodeSequence> > rules;

            NodeSequence matmul_ffn_seq = {{{-1, LAYER_ADD}, {-1, LAYER_MUL} }, false, "ffn"};
            NodeSequence matmul_ffn_backward_seq = {{{0, LAYER_MUL}, {0, LAYER_MUL}, {0, LAYER_ADD} }, true, "ffn"};
            rules[LAYER_MATMUL] = {matmul_ffn_seq, matmul_ffn_backward_seq};

            const int count   = (const int)structure->layers.size();
            for (int index = 0; index < count; index++) {
                auto layer = structure->layers[index];

                if (rules.find(layer->type) != rules.end()){
                    auto n = graph->getNodeByBlobName(layer->outputs[0]);

                    auto &seq_list = rules[layer->type];
                    for (auto &seq : seq_list) {
                        auto &lst = std::get<0>(seq);
                        bool reverse = std::get<1>(seq);
                        std::string label = std::get<2>(seq);

                        if (n->matchSequence(&lst[0], int(lst.size()), reverse)) {
                            layer->param->extra_config.insert(label);
                            // printf("layer %s is marked as %s\n", layer->name.c_str(), label.c_str());
                        } 
                    }
                }
            }

        }catch(...) {
            return Status(TNNERR_INST_ERR, "ContextMarker got exception");
        }

        return TNN_OK;
    }

}  // namespace context_marker

}  // namespace optimizer

}  // namespace TNN_NS
