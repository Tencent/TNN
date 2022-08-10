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

#ifndef TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_IR_H_
#define TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_IR_H_

#include <memory>
#include <string>
#include <tuple>
#include <map>
#include <set>
#include <ostream>
#include <vector>

#include "tnn/core/macro.h"
#include "tnn/interpreter/net_structure.h"
#include "tnn/core/layer_type.h"

namespace TNN_NS {

    struct Node;

    struct Edge {
        Edge(Node * _src, Node * _dst);
    public:
        Node * src;
        Node * dst;
    };

    struct Node {
        Node(std::shared_ptr<LayerInfo> &layer_info);
        // create placeholder node 
        Node(const std::string &blob_name);

        void addOutputEdge(Edge * e);
        void addInputEdge(Edge * e);
        void addInput(Edge * e);

        Node * prev(int id);
        Node * next(int id);

        bool matchSequence(std::pair<int, LayerType> * seq, int seq_len, bool reverse);

    public:
        std::string name;
        std::shared_ptr<LayerInfo> info;
        std::vector<Edge*> output_edges;
        std::vector<Edge*> input_edges;
    };

    struct Graph;

    struct HeirGraph;

    struct AnchorGraph;

    typedef std::function<std::shared_ptr<HeirGraph>(std::shared_ptr<AnchorGraph>)> graph_generator;

    struct Graph : public std::enable_shared_from_this<Graph> {

        Graph(std::vector<std::shared_ptr<LayerInfo> > layers);

        Graph(std::string proto_str);

        std::shared_ptr<Node> getNodeByBlobName(const std::string &blob_name);
        
        std::shared_ptr<Node> peekNodeByBlobName(const std::string &blob_name) const;

        void ConnectTwoNodes(Node * from, Node * to);

        bool rewrite(std::shared_ptr<Graph> &pattern, graph_generator generator);

        void dump(std::ostream &os) const;

        virtual std::vector<Node *> outputs() const {
            std::vector<Node *> res;
            for(auto &n : nodes) {
                if (n->output_edges.size() == 0) {
                    res.push_back(n.get());
                }
            }
            return res;
        }

        virtual std::vector<Node *> inputs() const {
            std::vector<Node *> res;
            for(auto &n : placeholders) {
                res.push_back(n.get());
            }
            return res;
        }

    public:
        std::vector<std::shared_ptr<Node>> nodes;
        std::vector<std::shared_ptr<Edge>> edges;
        std::vector<std::shared_ptr<Node>> placeholders;
        std::map<std::string, std::shared_ptr<Node>> blob_2_node;
    };


    // HeirGraph is generated from an AnchorGraph.
    // all edges should satisify: src and dst are in the HeirGraph
    // input and output edges for the HeirGraph are stored explicitly.
    // input nodes are not included in the nodes. e.g. the place holders
    // output nodes are included in the nodes. 
    struct HeirGraph : public Graph {
        std::vector<Node *> output_nodes;
        // std::vector<Edge *> input_edges;
        // std::vector<Edge *> output_edges;
        std::map<Node *, Node*> replace_map;

        HeirGraph(): Graph("") {};
        // Deep Copy of the nodes.
        HeirGraph(const AnchorGraph &g);

        void markOutput(Node *n);
        void markReplacement(Node *origin, Node * n);
        void markAllInOneNode(const AnchorGraph &g);

        void embed(std::shared_ptr<Graph> g, const std::shared_ptr<AnchorGraph> anchor, std::string name_prefx);

        virtual std::vector<Node *> outputs() const override {
            auto degree_zero_nodes = Graph::outputs();
            degree_zero_nodes.insert(degree_zero_nodes.end(), output_nodes.begin(), output_nodes.end());
            std::set<Node *> node_set(degree_zero_nodes.begin(), degree_zero_nodes.end());
            return std::vector<Node*>(node_set.begin(), node_set.end());
        }

    };


}

#endif // TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_IR_H_
