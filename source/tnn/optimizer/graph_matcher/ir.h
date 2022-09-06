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
#include "tnn/core/common.h"
#include "tnn/interpreter/net_structure.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/core/layer_type.h"
#include "tnn/optimizer/graph_matcher/common.h"

namespace TNN_NS {

    struct Node;

    struct Graph;

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
        Status updateInput(const std::string &name, const std::string &new_name, Edge * e);

        Status sanityCheck();

        std::shared_ptr<LayerInfo> info;
        std::vector<Edge*> output_edges;
        std::vector<Edge*> input_edges;

    };

    typedef std::function<std::shared_ptr<Graph>(std::shared_ptr<AnchorGraph>)> graph_generator;

    struct Graph : public std::enable_shared_from_this<Graph> {

        Graph() {};

        Graph(const std::vector<std::shared_ptr<Node>> _nodes, 
              const std::vector<std::shared_ptr<Node>> _placeholders, 
              const std::vector<std::shared_ptr<Edge>> _edges,
              const std::vector<std::shared_ptr<Tensor>> _tensors) 
              : nodes(_nodes), placeholders(_placeholders), edges(_edges), tensors(_tensors) 
        {
            RAISE_ON_ERROR(createUnspecifiedTensors());
        }

        Status fromInterpreted(NetStructure * , NetResource *);

        Graph(std::string proto_str);

        Status reBuildTensorIndex();

        Status RemoveDeadComponents();

        virtual Status sanityCheck();

        Status renameTensor(const std::string &old_name, const std::string &new_name);

        Status markOutput(const std::string &tensor_name);

        // will also handle the tensors
        Status addNode(const std::shared_ptr<Node> &pattern);

        // create node of specified type, Node name is set to the first output tensor_name, will also handle the tensors by addNode function
        Status createNode(const LayerType &type, const std::vector<std::string> &in_names, const std::vector<std::string> &out_names);

        const std::vector<std::weak_ptr<const Node>> allNodes() const;

        Status rewrite(std::shared_ptr<Graph> &pattern, graph_generator generator);

        void dump(std::ostream &os) const;

        // will create a placeholder node if tensor not found.
        std::shared_ptr<Node> getNodeOrCreatePlaceHolder(const std::string &tensor_name);

        std::shared_ptr<Node> getNodeByTensorName(const std::string &tensor_name) const;

        std::shared_ptr<Tensor> getTensorByName(const std::string &tensor_name) const;

        std::vector<const Tensor*> getTensorsByNames(const std::vector<std::string> &tensor_names) const ;

        virtual std::vector<Node*> outputNodes() const;
        // Returns the nodes that produce the input tensors. e.g. placeholders
        virtual std::vector<Node*> inputNodes() const;

        virtual std::vector<const Tensor*> outputs() const;
        virtual std::vector<const Tensor*> inputs() const;

        virtual Status setInputsOrder(std::vector<std::string> tensor_names);
        virtual Status setOutputsOrder(std::vector<std::string> tensor_names);

    protected:

        Status buildNodeTensorIndex(const std::shared_ptr<Node> );

        void embed(std::shared_ptr<Graph> g, const std::shared_ptr<AnchorGraph> anchor, std::string name_prefx) ;

        Status topologicalSort();

    private:
        Status createDefaultTensor(std::string name);
        Status createUnspecifiedTensors();

    protected:
        std::vector<std::shared_ptr<Node>> nodes;
        std::vector<std::shared_ptr<Edge>> edges;
        std::vector<std::shared_ptr<Node>> placeholders;
        std::vector<std::shared_ptr<Tensor>> tensors;

        std::set<std::string> marked_outputs;

        // following members are used to manage the ordering of outputs
        std::vector<std::string> output_order;

        // following members are managed by the reBuidlTensorIndex function
        std::unordered_map<std::string, std::shared_ptr<Tensor>> tensor_map;
        std::map<std::string, std::shared_ptr<Node>> tensor_2_node;
        std::map<std::string, std::vector<Edge*>> tensor_2_edge;

        NetStructure * tnn_structure = nullptr;
        NetResource * tnn_resource = nullptr;
    
    private:
        int rewrite_count = 0;

        friend class AnchorGraph;
    };

}

#endif // TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_IR_H_
