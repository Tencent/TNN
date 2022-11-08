#include <stdio.h>
#include <sstream>
#include <vector>
#include <fstream>

#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/graph_parser.h"
#include "tnn/optimizer/graph_matcher/text_graph_parser.h"
#include "tnn/optimizer/graph_matcher/graph_matcher.h"
#include "tnn/optimizer/graph_matcher/logger.h"
#include "tnn/optimizer/graph_matcher/graph_utils.h"
#include "tnn/optimizer/graph_matcher/graph_registry.h"

int main(int argc, char ** argv) {
    TNN_NS::Logger::instance().set_verbose_level("I");

    std::vector<std::string> text_graph = {
        "LayerNorm # some comments",
        "        MatMul<",
        "        Add",
        "                                      Mul<",
        "                            Mul<+>",
        "                Mul<        Add",
        "                Mul+>",
        "                Tanh@act",
        "        Mul     Add",
        "        Mul+>",
        "        MatMul",
        "        Add+{act}",
        "Add+>",
        "Add@branch",
        "Mul",
        "Mul+{branch}",
    };


    TNN_NS::GraphRegistry registry;

    std::shared_ptr<TNN_NS::Graph> graph, pattern;

    TNN_NS::TextGraphParser parser;
    TNN_NS::GraphParser graph_parser(&registry);

    auto status = parser.parseFromString(text_graph);
    if (status == TNN_NS::TNN_OK){
        graph = parser.getGraph();
        std::ofstream f("test.tnnproto");
        graph->dump(f);
    } else {
        printf("parse got error, code:%d msg:%s\n", int(status), status.description().c_str());
        return 0;
    }

    bool connected = false;
    RETURN_ON_FAIL(IsConnectedGraph(graph.get(), connected));
    if (!connected) {
        printf("The graph is not connected.\n");
        return 0;
    }

    {
        std::string graph_str = R"(
            graph(%a):
                %c = Add(%a)
                %d = Mul(%c)
                %e = Mul(%d, %c)
                return (%e)
        )";

        if (graph_parser.parseFromString(graph_str)) {
            pattern = graph_parser.getGraph();
            std::ofstream f("ssa_pattern.tnnproto");
            if (!pattern) {
                ERROR("invalid pattern");
                return -1;
            }
            pattern->dump(f);
        } else {
            return -1;
        }

        RETURN_ON_FAIL(registry.registerGraph("ssa", pattern));

        auto gen = [](std::shared_ptr<TNN_NS::AnchorGraph> in) -> std::shared_ptr<TNN_NS::Graph> {
            if (in->inputs().size() != 1 || in->outputs().size() != 1 ){
                return nullptr;
            }

            auto g = std::make_shared<TNN_NS::Graph>();
            auto in_name = "input_1";
            auto in1 = g->getNodeOrCreatePlaceHolder(in_name);
            auto status = g->createNode(TNN_NS::LAYER_SIGMOID, {in_name}, {"ssa_node"});
            if (status != TNN_NS::TNN_OK) {
                return nullptr;
            }

            return g;
        };

        graph->rewrite(pattern, gen);
    }

    {
        std::vector<std::string> text_graph_pattern = {
            "Add",
            "Mul@xxx",
            // "Add"
            // "MatMul Tanh",
            // "Add+>",
        };

        if (parser.parseFromString(text_graph_pattern)) {
            pattern = parser.getGraph();
            std::ofstream f("pattern.tnnproto");
            pattern->dump(f);
        }

        auto gen = [](std::shared_ptr<TNN_NS::AnchorGraph> in) -> std::shared_ptr<TNN_NS::Graph> {
            if (in->inputs().size() != 1 || in->outputs().size() != 1 ){
                return nullptr;
            }

            auto n_of_interest = in->getNodeByTensorName(std::string("@xxx"));
            if (!n_of_interest) {
                printf("roi node not found\n");
                return nullptr;
            }

            auto g = std::make_shared<TNN_NS::Graph>();
            auto in_name = "input_1";
            auto in1 = g->getNodeOrCreatePlaceHolder(in_name);
            auto status = g->createNode(TNN_NS::LAYER_TANH, {in_name}, {"new_heir_node"});
            if (status != TNN_NS::TNN_OK) {
                return nullptr;
            }

            return g;
        };

        graph->rewrite(pattern, gen);
    }


    {
        std::vector<std::string> text_graph_pattern = {
            "Add",
            "Mul    Mul<",
        };

        auto gen = [](std::shared_ptr<TNN_NS::AnchorGraph> in) -> std::shared_ptr<TNN_NS::Graph> {
            if (in->inputs().size() != 1 || in->outputs().size() != 1 ){
                return nullptr;
            }

            auto g = std::make_shared<TNN_NS::Graph>();
            auto in_name = "input_1";
            auto in1 = g->getNodeOrCreatePlaceHolder(in_name);
            auto status = g->createNode(TNN_NS::LAYER_LAYER_NORM, {in_name}, {"new_heir_norm"});
            if (status != TNN_NS::TNN_OK) {
                return nullptr;
            }

            return g;
        };

        if (parser.parseFromString(text_graph_pattern)) {
            pattern = parser.getGraph();
            std::ofstream f("pattern2.tnnproto");
            pattern->dump(f);
            graph->rewrite(pattern, gen);

            RETURN_ON_FAIL(registry.registerGraph("add_mul_mul", pattern));
        }

    }


    {
        std::string graph_str = R"(
            graph(%a, %b):
                %c = AnyType(%a)
                %d = AnyType(%c, %b)
                return (%d)
        )";

        auto gen = [](std::shared_ptr<TNN_NS::AnchorGraph> in) -> std::shared_ptr<TNN_NS::Graph> {
            if (in->inputs().size() != 2 || in->outputs().size() != 1 ){
                printf("Expect HeirGraph to Have 2 inputs and 1 outputs, but got %lu inputs and %lu outptus.\n",
                        in->inputs().size(), in->outputs().size());
                return nullptr;
            }

            auto g = std::make_shared<TNN_NS::Graph>();
            auto in_name = "input_1";
            auto in_name2 = "input_2";
            auto in1 = g->getNodeOrCreatePlaceHolder(in_name);
            auto in2 = g->getNodeOrCreatePlaceHolder(in_name2);
            auto status = g->createNode(TNN_NS::LAYER_CONVOLUTION, {in_name, in_name2}, {"_any_conv"});
            if (status != TNN_NS::TNN_OK) {
                return nullptr;
            }

            std::ofstream f("heir2.tnnproto");
            g->dump(f);

            return g;
        };

        if (graph_parser.parseFromString(graph_str)) {
            pattern = graph_parser.getGraph();
            std::ofstream f("pattern3.tnnproto");
            pattern->dump(f);
            graph->rewrite(pattern, gen);
        }
    }

    {
        std::string graph_str = R"(
            graph(%a):
                %e = Add(%a)
                %c, %d = add_mul_mul(%e)
                return (%c, %d)
        )";

        auto gen = [](std::shared_ptr<TNN_NS::AnchorGraph> in) -> std::shared_ptr<TNN_NS::Graph> {
            if (in->inputs().size() != 1 || in->outputs().size() != 1 ){
                printf("Expect HeirGraph to Have 1 inputs and 1 outputs, but got %lu inputs and %lu outptus.\n",
                        in->inputs().size(), in->outputs().size());
                return nullptr;
            }

            auto g = std::make_shared<TNN_NS::Graph>();
            auto in_name = "input_1";
            auto in1 = g->getNodeOrCreatePlaceHolder(in_name);
            auto status = g->createNode(TNN_NS::LAYER_CONVOLUTION, {in_name}, {"_ffn"});
            if (status != TNN_NS::TNN_OK) {
                return nullptr;
            }

            return g;
        };

        if (graph_parser.parseFromString(graph_str) == TNN_NS::TNN_OK) {
            pattern = graph_parser.getGraph();
            std::ofstream f("ssa_with_function_graph.tnnproto");
            pattern->dump(f);

            if (graph) graph->rewrite(pattern, gen);
        }
    }

    if (graph) {
        std::ofstream f("rewrited.tnnproto");
        graph->dump(f);
    }

    return 0;
}
