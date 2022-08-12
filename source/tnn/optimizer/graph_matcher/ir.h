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
#include "tnn/core/status.h"
#include "tnn/interpreter/net_structure.h"
#include "tnn/core/layer_type.h"

#define RAISE_ON_ERROR(status)                                  \
    do {                                                        \
        auto _status = status;                                  \
        if ((_status) != TNN_OK) {                              \
            throw std::runtime_error(_status.description());    \
        }                                                       \
    } while (0)

namespace TNN_NS {

    struct Node;

    struct Graph;

    struct HeirGraph;

    struct AnchorGraph;

    struct Tensor {
        Tensor(const std::string &_name): name(_name) {}
        std::string name;
        DataType data_type;
        DataFormat format;
        DimsVector dims;
    };

    struct Edge {
        Edge(Node * _src, Node * _dst, const std::string &tensor);
        Node * src;
        Node * dst;
        std::string tensor_name;
    };

    struct Node {

        Node(std::shared_ptr<LayerInfo> &layer_info);
        // create placeholder node 
        Node(const std::string &tensor_name);

        std::string name() const {return info->name;}

        Status addOutputEdge(Edge * e);
        Status addInputEdge(Edge * e);
        Status addInput(Edge * e);

        Status sanityCheck();

        std::shared_ptr<LayerInfo> info;
        std::vector<Edge*> output_edges;
        std::vector<Edge*> input_edges;

    };

    typedef std::function<std::shared_ptr<HeirGraph>(std::shared_ptr<AnchorGraph>)> graph_generator;

    struct Graph : public std::enable_shared_from_this<Graph> {

        Graph() {};

        Graph(const std::vector<std::shared_ptr<Node>> _nodes, 
              const std::vector<std::shared_ptr<Node>> _placeholders, 
              const std::vector<std::shared_ptr<Edge>> _edges) 
              : nodes(_nodes), placeholders(_placeholders), edges(_edges) {}

        Status fromNetStructure(std::vector<std::shared_ptr<LayerInfo> > layers);

        Graph(std::string proto_str);

        Status reBuildTensorIndex();

        Status RemoveDeadComponents();

        Status sanityCheck();

        Status renameTensor(const std::string &old_name, const std::string &new_name);

        Status markOutput(const std::string &tensor_name);

        Status addNode(const std::shared_ptr<Node> &pattern);

        const std::vector<std::weak_ptr<const Node>> allNodes() const;

        Status rewrite(std::shared_ptr<Graph> &pattern, graph_generator generator);

        void dump(std::ostream &os) const;

        // will create a placeholder node if tensor not found.
        std::shared_ptr<Node> getNodeOrCreatePlaceHolder(const std::string &tensor_name);

        std::shared_ptr<Node> getNodeByTensorName(const std::string &tensor_name) const;

        std::shared_ptr<Tensor> getTensorByName(const std::string &tensor_name) const;

        std::vector<const Tensor*> getTensorsByNames(const std::vector<std::string> &tensor_names) const throw(...);

        virtual std::vector<Node*> outputs() const {
            std::vector<Node *> res;
            for(auto &n : nodes) {
                if (n->output_edges.size() == 0) {
                    res.push_back(n.get());
                }
            }
            return res;
        }

        virtual std::vector<Node*> inputs() const {
            std::vector<Node*> res;
            for(auto &n : placeholders) {
                res.push_back(n.get());
            }
            return res;
        }

        virtual std::vector<const Tensor*> outputs_() const;
        virtual std::vector<const Tensor*> inputs_() const;

    protected:

        Status buildNodeTensorIndex(const std::shared_ptr<Node> );

    protected:
        std::vector<std::shared_ptr<Node>> nodes;
        std::vector<std::shared_ptr<Edge>> edges;
        std::vector<std::shared_ptr<Node>> placeholders;

        std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;
        std::set<std::string> marked_outputs;

        std::map<std::string, std::shared_ptr<Node>> tensor_2_node;
        std::map<std::string, std::vector<Edge*>> tensor_2_edge;
    
    private:
        int rewrite_count = 0;

        friend class AnchorGraph;
        friend class HeirGraph;
    };


    // HeirGraph is generated from an AnchorGraph.
    // all edges should satisify: src and dst are in the HeirGraph
    // input and output edges for the HeirGraph are stored explicitly.
    // input nodes are not included in the nodes. e.g. the placeholders
    // output nodes are included in the nodes. 
    struct HeirGraph : public Graph {
        std::vector<Node *> output_nodes;
        std::map<Node *, Node*> replace_map;

        HeirGraph(): Graph("") {};
        // Deep Copy of the nodes.
        HeirGraph(const AnchorGraph &g);

        void markReplacement(Node *origin, Node * n);
        void markAllInOneNode(const AnchorGraph &g);

        void embed(std::shared_ptr<Graph> g, const std::shared_ptr<AnchorGraph> anchor, std::string name_prefx) throw(...);

        virtual std::vector<Node *> outputs() const override {
            auto degree_zero_nodes = Graph::outputs();
            degree_zero_nodes.insert(degree_zero_nodes.end(), output_nodes.begin(), output_nodes.end());
            std::set<Node *> node_set(degree_zero_nodes.begin(), degree_zero_nodes.end());
            return std::vector<Node*>(node_set.begin(), node_set.end());
        }

    };


}

#endif // TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_IR_H_
