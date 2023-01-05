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

#include "tnn/optimizer/graph_matcher/text_graph_parser.h"

#include <string>
#include <sstream>
#include <list>
#include <iostream>
#include <fstream>

#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/optimizer/graph_matcher/logger.h"

namespace TNN_NS {
void TextGraphParser::parseComments() {
    while(true) {
        Token cur = l_.next();
        switch (cur.kind) {
            case TK_EOF:
            case TK_WHITESPACE_EOF:
            case TK_NEWLINE:
                return;
            default:
                break;
        }
    }
    return;
}

int getTokenOffset(const Token &tk) {
    int of = tk.str.offset();
    int i = -1;
    while(of + i > 0 && tk.str[i] != '\n')
        i--;
    return - i -1;
}

void TextGraphParser::parseNode(const Token &layer_tok) {
    int offset = getTokenOffset(layer_tok);
    DEBUG("parsed Node [%s] at offset:%d", layer_tok.text().c_str(), offset);

    TextNode n(offset);
    n.source = layer_tok;

    auto finalizeInput = [&](int &shift, const Token &tok) {
        TextInput in(shift); 
        in.source = tok;
        n.inputs.push_back(in);
        shift = 0;
    };

    auto parseNameRef = [&](const int &shift, const Token &cur) {
        if (shift != 0) unexpect(cur);
        expect(l_.lookahead(), TK_IDENT);
        TextInput in(l_.lookahead().text());
        in.source = l_.lookahead();
        n.inputs.push_back(std::move(in));
        l_.next();
        expect(l_.lookahead(), '}');
        l_.next();
    };

    int shift = 0;
    while(true) {
        auto cur = l_.cur();
        auto &tk = cur;
        DEBUG("token kind:%d[%15s] text:\"%s\"", tk.kind, tk.name().c_str(), tk.text().c_str());
        switch (cur.kind) {
            case TK_EOF:
            case TK_WHITESPACE_EOF:
            case TK_NEWLINE:
            case TK_WHITESPACE:
            case '#':
                if (l_.prev().kind != '}')
                    finalizeInput(shift, l_.prev());
                g_.nodes.push_back(std::move(n));
                return;
            case '<':
                if (shift != 0) unexpect(cur);
                shift -= 1;
                while(l_.lookahead().kind == '<') {
                    l_.next();
                    shift -= 1;
                }
                break;
            case '>':
                if (shift != 0) unexpect(cur);
                shift += 1;
                while(l_.lookahead().kind == '>') {
                    l_.next();
                    shift += 1;
                }
                break;
            case '+':
                finalizeInput(shift, l_.prev());
                break;
            case '@':
                finalizeInput(shift, l_.prev());
                expect(l_.lookahead(), TK_IDENT);
                n.name = l_.lookahead().text();
                l_.next();
                l_.next();
                g_.nodes.push_back(std::move(n));
                return;
            case '{':
                parseNameRef(shift, cur);
                break;
            default:
                unexpect(cur);
                break;
        }
        l_.next();
    }
    throw std::runtime_error("Error, code is not expected to be here.");
}

void TextGraphParser::parseLine() {
    DEBUG("new line ----------------------------------------------");
    while(true) {
        Token cur = l_.next();
        switch (cur.kind) {
            case TK_EOF:
            case TK_WHITESPACE_EOF:
            case TK_NEWLINE:
                return;
            case '#':
                parseComments();
                return;
            case TK_WHITESPACE:
                break;
            default:
                expect(cur, TK_LAYER_TYPE);
                parseNode(cur);
                break;
        }
    }
    throw std::runtime_error("Error, parseLine function is not expected to be here.");
}


Status TextGraphParser::parseFromString(std::string text_graph) {
    l_ = Lexer(text_graph);
    g_ = TextGraph();

    try {

        while (l_.cur().kind != TK_EOF) {
            parseLine();
            g_.nodes.push_back(NEW_LINE_OFFSET);
        }

        // those codes are used to dump the parsed text graph for debug
        // for(auto n: g_.nodes) {
        //     for (int i=0;i<n.offset;i++) printf(" ");
        //     printf("%s", n.source.text().c_str());
        //     if (n.name.length() > 0) {
        //         printf("@%s", n.name.c_str());
        //     }
        //     for(auto input : n.inputs) {
        //         if (input.tag == REF_BY_SHIFT)
        //             printf(" [%d]", input.input.as_shift);
        //         else 
        //             printf(" [%s]", input.input.as_name.c_str());
        //     }
        //     printf("\n");
        // }

        graph_ = std::make_shared<Graph>("");

        RETURN_ON_NEQ(constructGraph(g_, graph_.get()), TNN_OK);

    } catch (const std::runtime_error& error) {
        ERROR("%s", error.what());
        return Status(TNNERR_COMMON_ERROR, error.what());
    } catch (...) {
        ERROR("Parser got unknow error.");
        return Status(TNNERR_COMMON_ERROR, "Parser got unknow error.");
    }

    return TNN_OK;
}

struct Slot{
    int offset;
    Node * node;
    Slot(){};
    Slot(Node *n, int of) : offset(of), node(n) {}
};

struct SlotManager {

    std::list<Slot> slots;
    std::map<std::string, Slot> named_slots;
    std::vector<std::shared_ptr<Node>> place_holders;

    void insertSlot(const Slot &slot) {
        auto it = slots.begin();
        for(; it != slots.end(); it++) {
            bool equal =  it->offset == slot.offset;
            bool next = it->offset < slot.offset;
            if (next) {
                continue;
            } else if (equal) {
                DEBUG("Update Slot[%d][%s]", slot.offset, slot.node->name().c_str());
                for(auto &name : slot.node->info->outputs) named_slots[name] = slot;
                it->node = slot.node;
                return; 
            } else {
                break;
            }
        }
        if (it != slots.end())
            DEBUG("insert Slot[%d]<%s> before [%d]", slot.offset, slot.node->name().c_str(), it->offset);
        else 
            DEBUG("insert Slot[%d]<%s> at end", slot.offset, slot.node->name().c_str());

        for(auto &name: slot.node->info->outputs) named_slots[name] = slot;
        slots.insert(it, slot);
    };

    void touchPlaceHolder(const int &offset) {
        auto it = slots.begin();
        for(; it != slots.end(); it++) {
            if (it->offset < offset) {
                continue;
            } else if (it->offset == offset) {
                return; 
            } else {
                break;
            }
        }
        DEBUG("Create Placeholder at %d", offset);
        auto n = std::make_shared<Node>(std::string("placeholder_") + std::to_string(place_holders.size()));
        Slot s(n.get(), offset);
        insertSlot(s);
        place_holders.push_back(n);
    }

    bool getSlot(const int &offset, int shift, Slot * slot) {
        std::list<Slot>::iterator pos = slots.begin();
        while(pos != slots.end() && offset != pos->offset) pos++;
        while(shift != 0 && pos !=slots.end()) {
            if (shift > 0)  {
                shift--;
                pos++;
                if (pos == slots.end() ) {
                    return false;
                }
            }
            if (shift < 0) {
                shift ++;
                if (pos == slots.begin() ) {return false;}
                pos--;
            }
        }
        *slot = *pos;
        return true;
    };


    bool getSlotByName(const std::string &name, Slot * slot) {
        if (named_slots.find(name) == named_slots.end()) {
            return false;
        }
        *slot = named_slots[name];
        return true;
    };

};



Status constructGraph(const TextGraph &tg, Graph * graph) 
{
    SlotManager manager;

    std::map<std::string, int> node_cnt;
    int tensor_cnt = 0;
    auto getNodeName = [&](const Token &tk) -> std::string {
        if (node_cnt.count(tk.text()) == 0) node_cnt[tk.text()] = 0;
        return tk.text() + std::string("_") + std::to_string(node_cnt[tk.text()]++);
    };
    auto getTensorName = [&]() -> std::string {
        return std::to_string(tensor_cnt++);
    };

    std::vector<std::shared_ptr<Edge>> edges;
    std::vector<std::shared_ptr<Node>> nodes;

    auto createNode = [&](const TextNode & text_node) -> std::shared_ptr<Node> {
        auto n = std::make_shared<Node>(getNodeName(text_node.source));
        n->info->type = GlobalConvertLayerType(text_node.source.text());

        if (n->info->type != LAYER_PLACEHOLDER) {
            n->info->outputs = {getTensorName()};
            if (text_node.name.length() > 0 ) {
                n->info->outputs = {text_node.name};
            }

            for(auto &input : text_node.inputs) {
                Slot s;
                bool ok=false;
                switch (input.tag) {
                    case REF_BY_SHIFT:
                        ok = manager.getSlot(text_node.offset, input.input.as_shift, &s);
                        break;
                    case REF_BY_NAME:
                        ok = manager.getSlotByName(input.input.as_name, &s);
                        break;
                }
                if (!ok) {
                    reportError("specified input not found when constructiing graph.", input.source);
                }
                auto e = std::make_shared<Edge>(s.node, n.get(), s.node->info->outputs[input.index]);
                edges.push_back(e);
                RAISE_ON_ERROR(s.node->addOutputEdge(e.get()));
                RAISE_ON_ERROR(n->addInput(e.get()));
            }
        }

        nodes.push_back(n);
        return n;
    };

    size_t line_s = 0;
    for(;line_s<tg.nodes.size();) {
        size_t line_e = line_s;
        while(line_e<tg.nodes.size() && tg.nodes[line_e].offset != NEW_LINE_OFFSET) line_e++;

        for(auto i=line_s;i<line_e;i++) {
            manager.touchPlaceHolder(tg.nodes[i].offset);
        }

        std::vector<std::shared_ptr<Node>> line;
        for(auto i=line_s;i<line_e;i++) {
            auto n = createNode(tg.nodes[i]);
            line.push_back(n);
        }

        int cnt = 0;
        for(size_t i=line_s;i<line_e;i++, cnt++) {
            manager.insertSlot(Slot(line[cnt].get(), tg.nodes[i].offset));
        }
        line_s = line_e + 1;
    }

    std::vector<std::shared_ptr<Node>> placeholders = manager.place_holders;
    // User inputed placeholder is stored in the nodes.
    placeholders.insert(placeholders.end(), nodes.begin(), nodes.end());
    placeholders.erase(std::remove_if(placeholders.begin(), 
                                      placeholders.end(),
                                        [](std::shared_ptr<Node> &n){
                                            return n->output_edges.size() == 0 || n->info->type != LAYER_PLACEHOLDER;
                                        }),
                                    placeholders.end());

    nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                                    [](std::shared_ptr<Node> &n){ return  n->info->type == LAYER_PLACEHOLDER; }),
                                    nodes.end());
                                
    *graph = Graph(nodes, placeholders, edges, {});
    RAISE_ON_ERROR(graph->reBuildTensorIndex());

    std::vector<std::string> in_order, out_order;

    for(auto t : graph->inputs()) in_order.push_back(t->name);
    for(auto t : graph->outputs()) out_order.push_back(t->name);
    RAISE_ON_ERROR(graph->setInputsOrder(in_order));
    RAISE_ON_ERROR(graph->setOutputsOrder(out_order));
    
    return TNN_OK;
}

} // namespace tnn