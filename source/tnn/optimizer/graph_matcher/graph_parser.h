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

#ifndef TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_PARSER_H_
#define TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_PARSER_H_

#include <vector>
#include <memory>
#include <sstream>
#include <string>

#include "tnn/core/macro.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/lexer.h"
#include "tnn/optimizer/graph_matcher/graph_registry.h"

namespace TNN_NS {

struct Value {
    Value(const std::string &ident, const Token& tk): identifier(ident), source(tk) {};
    std::string identifier;
    Token source;

    bool operator<(const  Value &o) const {
        return identifier < o.identifier;
    }
};

struct Attributes {
    std::map<Token, Token> kv;
};

struct SSANode {
    std::vector<Value> inputs;
    std::vector<Value> outputs;
    Attributes attrs;
    Token source;

};

struct SSAGraph {
    std::vector<Value> inputs;
    std::vector<SSANode> nodes;
};

struct GraphParser {

    GraphParser() : l_(SubStr("")) {}
    GraphParser(GraphRegistry * registry) : l_(SubStr(""), registry), registry_(registry) {}

    Status parseFromString(std::string text_graph);

    std::shared_ptr<Graph> getGraph() {
        return graph_;
    }

protected:

    void parseComments();
    void parseNode();
    void parseReturn();
    void parseValue(std::vector<Value> &container);

    void parseLine();


private:
    Lexer l_;
    SSAGraph g_;

    GraphRegistry * registry_ = nullptr;

    std::shared_ptr<Graph> graph_ = nullptr;

};

}

#endif // TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_PARSER_H_
