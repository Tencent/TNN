#include "tnn/optimizer/graph_matcher/graph_parser.h"

#include <string>
#include <sstream>
#include <list>
#include <iostream>
#include <fstream>

#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/graph_registry.h"
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
    expect(layer_type, {TK_LAYER_TYPE, TK_GRAPH_FUNCTION});
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

Status constructGraph(const SSAGraph &ssa, Graph * graph, GraphRegistry * registry);

Status GraphParser::parseFromString(std::string graph_str) {
    try {
        l_ = Lexer(graph_str, registry_);
        g_ = SSAGraph();

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

        RETURN_ON_NEQ(constructGraph(g_, graph_.get(), registry_), TNN_OK);

    } catch (const std::runtime_error& error) {
        ERROR("%s", error.what());
        return Status(TNNERR_COMMON_ERROR, error.what());
    } catch (...) {
        ERROR("Parser got unknow error.");
        return Status(TNNERR_COMMON_ERROR, "Parser got unknow error.");
    }

    return TNN_OK;
}

Status constructGraph(const SSAGraph &ssa, Graph * graph, GraphRegistry * registry) {

    std::vector<std::shared_ptr<Edge>> edges;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<std::shared_ptr<Tensor>> tensors;
    std::vector<std::shared_ptr<Node>> placeholders;

    std::vector<std::string> input_order, output_order;

    std::map<std::string, Node *> tensor_2_node;

    std::map<std::string, int> node_cnt;
    auto getNodeName = [&](const Token &tk) -> std::string {
        if (node_cnt.count(tk.text()) == 0) node_cnt[tk.text()] = 0;
        return tk.text() + std::string("_") + std::to_string(node_cnt[tk.text()]++);
    };

    auto createNode = [&](const SSANode& node) -> std::shared_ptr<Node> {
        if (node.outputs.size() == 0) {
            reportError("Node has no output", node.source);
        }

        auto n = std::make_shared<Node>(getNodeName(node.source));
        n->info->type = GlobalConvertLayerType(node.source.text());
        n->info->type_str = node.source.text();

        if (n->info->type == LAYER_PLACEHOLDER) {
            reportError("placeHolder is not expected", node.source);
        }

        for(auto &input : node.inputs) {
            if (tensor_2_node.count(input.identifier) == 0) {
                reportError("specified input not found when constructiing graph.", input.source);
            }
            Node * src = tensor_2_node.at(input.identifier);
            auto e = std::make_shared<Edge>(src, n.get(), input.identifier);
            edges.push_back(e);
            RAISE_ON_ERROR(src->addOutputEdge(e.get()));
            RAISE_ON_ERROR(n->addInput(e.get()));
        }

        n->info->outputs = {};
        for(auto &v : node.outputs) {
            n->info->outputs.push_back(v.identifier);
            tensor_2_node[v.identifier] = n.get();
        }

        nodes.push_back(n);
        return n;
    };

    std::map<std::string, size_t> ssa_cnt_map;
    auto get_ssa_cnt = [&](const std::string name) -> size_t {
        if (ssa_cnt_map.count(name) > 0) {
            size_t cnt = ssa_cnt_map.at(name);
            ssa_cnt_map[name] = cnt + 1;
            return cnt + 1;
        }
        ssa_cnt_map[name] = 0;
        return 0;
    };
    auto createFunction = [&](const SSANode& node) -> void {
        if (!registry) {
            ERRORV("GraphRegistry is nullptr.", msg);
            reportError(msg, node.source);
        }
        auto g = registry->queryGraphByName(node.source.text());
        if (!g) {
            ERRORV("GraphFunction with name:%s is found.", msg, node.source.text().c_str());
            reportError(msg, node.source);
        }
        if (node.inputs.size() != g->inputs().size()) {
            ERRORV("GraphFunction input size got %lu, expected: %lu.", msg, node.inputs.size(), g->inputs().size());
            reportError(msg, node.source);
        }
        if (node.outputs.size() != g->outputs().size()) {
            ERRORV("GraphFunction output size got %lu, expected: %lu.", msg, node.outputs.size(), g->outputs().size());
            reportError(msg, node.source);
        }

        std::shared_ptr<Graph> sub_graph = g->Copy();
        std::string name_prefix = node.source.text() + std::string("_") + std::to_string(get_ssa_cnt(node.source.text())) + std::string("_");
        for(auto &t: sub_graph->tensors) { 
            // DEBUG("rename subgraph tensor from %s to %s", t->name.c_str(), (name_prefix+t->name).c_str());
            sub_graph->renameTensor(t->name, name_prefix + t->name);
        }

        for(size_t i=0;i<node.inputs.size();i++) {
            auto input = node.inputs[i];
            if (tensor_2_node.count(input.identifier) == 0) {
                reportError("specified input not found when constructiing graph.", input.source);
            }

            Node * src = tensor_2_node.at(input.identifier);
            const Tensor * t = sub_graph->inputs()[i];

            // DEBUG("procesing %lu th input of name:%s GraphFunction input name:%s ",i, input.identifier.c_str(), t->name.c_str());
            // DEBUG("\tsrc %p type:%s of name:%s", src, layerTypeName(src->info->type).c_str(), src->name().c_str());
            // DEBUG("\trename tensor from %s to %s", t->name.c_str(), input.identifier.c_str());

            RAISE_ON_ERROR(sub_graph->renameTensor(t->name, input.identifier));

            auto edges = sub_graph->tensor_2_edge.at(t->name);
            for(auto e : edges) {
                e->src->output_edges.erase(std::remove_if(e->src->output_edges.begin(), e->src->output_edges.end(), [&](Edge * cur){
                                            return cur == e;
                                        }), e->src->output_edges.end());
                e->src = src;
                RAISE_ON_ERROR(src->addOutputEdge(e));

            }
        }

        for(size_t i=0;i<node.outputs.size();i++) {
            auto output = node.outputs[i];
            const Tensor * t = sub_graph->outputs()[i];

            // DEBUG("procesing %lu th output of name:%s GraphFunction output name:%s ",i, output.identifier.c_str(), t->name.c_str());
            // DEBUG("\trename tensor %p from %s to %s, ouput_order.size()=%lu", t, t->name.c_str(), output.identifier.c_str(), sub_graph->output_order.size());

            RAISE_ON_ERROR(sub_graph->renameTensor(t->name, output.identifier));

            auto n = sub_graph->getNodeByTensorName(output.identifier);
            if (!n) {
                ERRORV("GraphFunction output node of name %s not found.", msg, output.identifier.c_str());
                reportError(msg, node.source);
            }
            tensor_2_node[output.identifier] = n.get();
        }

        for(auto n : sub_graph->nodes) {
            n->info->name = getNodeName(Token(TK_LAYER_TYPE, layerTypeName(n->info->type)));
        }

        nodes.insert(nodes.begin(), sub_graph->nodes.begin(), sub_graph->nodes.end());
        edges.insert(edges.begin(), sub_graph->edges.begin(), sub_graph->edges.end());
        tensors.insert(tensors.begin(), sub_graph->tensors.begin(), sub_graph->tensors.end());
    };

    for(auto &v : ssa.inputs) {
        auto n = std::make_shared<Node>(v.identifier);
        placeholders.push_back(n);
        tensor_2_node[v.identifier] = n.get();
        input_order.push_back(v.identifier);
    }

    bool on_exit = false;
    std::set<Value> return_values;

    for(auto &ssa_node : ssa.nodes) {
        if (on_exit) {
            reportError("unexpected Node", ssa_node.source);
        }
        if (ssa_node.source.kind == TK_RETURN) {
            for(auto &v : ssa_node.inputs) {
                return_values.insert(v);
                output_order.push_back(v.identifier);
            }
            on_exit = true;
            continue;
        }
        if (ssa_node.source.kind == TK_GRAPH_FUNCTION) {
            createFunction(ssa_node);
        } else {
            createNode(ssa_node);
        }
    }

    *graph = Graph(nodes, placeholders, edges, tensors);
    RAISE_ON_ERROR(graph->reBuildTensorIndex());
    for(auto v : return_values) {
        RAISE_ON_ERROR(graph->markOutput(v.identifier));
    }
    RAISE_ON_ERROR(graph->reBuildTensorIndex());
    RAISE_ON_ERROR(graph->setInputsOrder(input_order));
    RAISE_ON_ERROR(graph->setOutputsOrder(output_order));

    for(auto &v : return_values) {
        if (tensor_2_node.count(v.identifier) == 0) {
            reportError("specified return value not found when constructiing graph.", v.source);
        }
        RAISE_ON_ERROR(graph->markOutput(v.identifier));
    }

    return TNN_OK;
}

}