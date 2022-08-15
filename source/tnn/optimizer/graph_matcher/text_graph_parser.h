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

#ifndef TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_TEXT_PARSER_H_
#define TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_TEXT_PARSER_H_

#include <vector>
#include <memory>
#include <sstream>
#include <string>

#include "tnn/core/macro.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/lexer.h"

// TODO:
//      1. Suppport Node that have multiple outputs.
//      2. Register a graph as a node , for later reference.

namespace TNN_NS {

enum InputType {
    REF_BY_SHIFT = 0,
    REF_BY_NAME = 1,
};

const int NEW_LINE_OFFSET=-0x7fff;

union Input {
    int as_shift;
    std::string as_name; 
    Input(): as_name(std::string()){}
    ~Input() {}
};

struct TextInput {
    TextInput(int shift = 0):  tag(REF_BY_SHIFT) {input.as_shift = shift;};
    TextInput(std::string name):  tag(REF_BY_NAME) {input.as_name = name;};
    TextInput(const TextInput &rhs) {
        source = rhs.source;
        tag = rhs.tag; 
        switch (tag) {
            case REF_BY_NAME:
                input.as_name = rhs.input.as_name;
                break;
            case REF_BY_SHIFT:
                input.as_shift = rhs.input.as_shift;
                break;
        }
    }
    Input input;
    InputType tag;
    Token source;
    int index = 0;
};

struct TextNode {
    TextNode(int offset): offset(offset) {}
    TextNode(int offset, std::string name): offset(offset), name(name) {}

    std::string name;
    int offset;
    Token source;
    std::vector<TextInput> inputs;
};

struct TextGraph {
    TextGraph() {}
    bool ConstructGraph(Graph * graph);

    std::vector<TextNode> nodes;
};

Status constructGraph(const TextGraph &tg, Graph * graph) ;

struct TextGraphParser {

    TextGraphParser() : l_(SubStr("")) {}

    Status parseFromString(std::string text_graph);
    Status parseFromString(std::vector<std::string> text_graph) {
        std::stringstream builder;
        for(auto &s : text_graph) {
            builder<<s<<"\n";
        }
        return parseFromString(builder.str());
    };

    std::shared_ptr<Graph> getGraph() {
        return graph_;
    }

protected:

    void parseComments();
    void parseNode(const Token &layer_tok);

    void parseLine();


private:
    Lexer l_;
    TextGraph g_;

    std::shared_ptr<Graph> graph_ = nullptr;

};

}

#endif // TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_TEXT_PARSER_H_
