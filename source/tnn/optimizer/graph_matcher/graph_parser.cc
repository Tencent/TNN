#include "tnn/optimizer/graph_matcher/graph_parser.h"

#include <string>
#include <sstream>
#include <list>
#include <iostream>
#include <fstream>

#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/logger.h"

namespace TNN_NS {


void GraphParser::parseComments() {
    Token first = l_.next();
    expect(first, '#');
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

void GraphParser::parseValue(std::vector<Value> &container) {
    bool after_comma = false;
    while (true) {
        Token cur = l_.cur();
        DEBUG("\t\tparseValue got token kind:%d[%15s] text:\"%s\"", cur.kind, cur.name().c_str(), cur.text().c_str());
        switch (cur.kind) {
            case ')':
            case '{':
            case '=':
            case TK_LAYER_TYPE:
                if (after_comma) unexpect(cur);
                return;
            case TK_WHITESPACE:
                l_.next();
                break;
            case ',':
                after_comma = true;
                l_.next();
                break;
            case '%':
                cur = l_.lookahead();
                if (cur.kind != TK_IDENT && cur.kind != TK_NUMBER) {
                    unexpect(cur);
                }
                container.push_back(Value(cur.text(),  cur));
                DEBUG("Add value[%s]", cur.text().c_str());
                l_.next(); 
                l_.next();
                after_comma = false;
                break;
            default :
                unexpect(cur);
                return;
        }
    }
};

void GraphParser::parseNode() {

    SSANode n;

    Token cur = l_.cur();
    DEBUG("now parse Node, got kind:%s\n", cur.name().c_str());
    expect(cur, '%');
    parseValue(n.outputs);

    expect(l_.next(), '=');
    while(l_.cur().kind == TK_WHITESPACE) l_.next();

    Token layer_type = l_.next();
    expect(layer_type, TK_LAYER_TYPE);
    n.source = layer_type;

    expect(l_.next(), '(');
    parseValue(n.inputs);
    expect(l_.next(), ')');

    while(true) {
        Token cur = l_.cur();

        switch (cur.kind) {
            case TK_NEWLINE:
            case '#':
                g_.nodes.push_back(n);
                DEBUG("Add Node[%s]", n.source.text().c_str());
                return;
            case TK_WHITESPACE:
                l_.next();
                break;
            default:
                unexpect(cur);
        }
    }
}

void GraphParser::parseReturn() {
    SSANode n;

    Token cur = l_.next();
    expect(cur, TK_RETURN);
    n.source = cur;

    while(l_.cur().kind == TK_WHITESPACE) l_.next();

    expect(l_.next(), '(');
    parseValue(n.inputs);
    expect(l_.next(), ')');

    g_.nodes.push_back(n);
    DEBUG("Add return node [%s]", n.source.text().c_str());
}

void GraphParser::parseLine() {
    bool on_exit = false;
    while(true) {
        Token cur = l_.cur();
        DEBUG("\t\tparseLine get token kind:%d[%15s] text:\"%s\"", cur.kind, cur.name().c_str(), cur.text().c_str());
        switch (cur.kind) {
            case TK_EOF:
            case TK_WHITESPACE_EOF:
            case TK_NEWLINE:
                l_.next();
                return;
            case TK_RETURN:
                if (on_exit) unexpect(cur);
                on_exit =true;
                parseReturn();
                return;
            case '#':
                parseComments();
                return;
            case TK_WHITESPACE:
                l_.next();
                break;
            default:
                if (on_exit) unexpect(cur);
                on_exit =true;
                expect(cur, '%');
                parseNode();
                break;
        }
    }
}

Status constructGraph(const SSAGraph &ssa, Graph * graph);

Status GraphParser::parseFromString(std::string graph_str) {
    l_ = Lexer(graph_str);
    g_ = SSAGraph();

    try {
        // parse graph header and inputs
        while(l_.cur().kind == TK_WHITESPACE || l_.cur().kind == TK_NEWLINE) l_.next();
        expect(l_.next(), TK_GRAPH);
        expect(l_.next(), '(');
        parseValue(g_.inputs);
        expect(l_.next(), ')');
        expect(l_.next(), ':');

        while (l_.cur().kind != TK_EOF) {
            parseLine();
        }

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

Status constructGraph(const SSAGraph &ssa, Graph * graph) {

    std::vector<std::shared_ptr<Edge>> edges;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<std::shared_ptr<Node>> placeholders;
    std::vector<std::shared_ptr<Node>> outputs;

    for(auto &n : ssa.inputs) {
        placeholders.push_back(std::make_shared<Node>(n.identifier));
    }

    for(auto &n : ssa.outputs) {
        auto dummy = std::make_shared<Node>(n.identifier);
        dummy->info->type = LAYER_DUMMY_TYPE;
        outputs.push_back(dummy);
    }

    return TNN_OK;
}

}