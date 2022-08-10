#include <stdio.h>
#include <sstream>
#include <vector>
#include <fstream>

#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/parser.h"
#include "tnn/optimizer/graph_matcher/graph_matcher.h"
#include "tnn/optimizer/graph_matcher/logger.h"

int main(int argc, char ** argv) {
    // tnn::Logger::instance().set_verbose_level("D");


    std::vector<std::string> text_graph = {
        // "Add@ff",
        // "   Conv<+{ff} Const# ",
        // "            Conv>+<                  # this is nor a line",
        // "Add+>",
        // "LayerNorm",
        // "    MatMul<",
        // "    Add",
        // "           Mul<@2    Mul<<",
        // "           Mul+<#2",
        // "    Mul    Add{2}+>",
        // "    Mul+>",
        // "    Tanh",
        // "    Add",
        // "    Mul+>>",
        // "Add",
"LayerNorm",
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
"Add",
"Mul     ",
"Mul",
    };


    tnn::TextGraphParser parser;
    bool ok = parser.parseFromString(text_graph);

    std::shared_ptr<tnn::Graph> graph, pattern;

    if (ok) {
        graph = parser.getGraph();
        std::ofstream f("test.tnnproto");
        graph->dump(f);
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

        auto gen = [](std::shared_ptr<tnn::AnchorGraph> in) -> std::shared_ptr<tnn::HeirGraph> {
            if (in->inputs().size() != 1 || in->outputs().size() != 1 ){
                return nullptr;
            }

            auto n_of_interest = in->peekNodeByBlobName(std::string("@xxx"));
            if (!n_of_interest) {
                return nullptr;
            }

            auto g = std::make_shared<tnn::HeirGraph>();
            int num_inputs = 1;
            for(int i=0;i<num_inputs;i++) {
                auto ph = g->getNodeByBlobName(std::string("PlaceHolder_") + std::to_string(i));
                ph->info->type = tnn::LAYER_NOT_SUPPORT;
            }
            auto new_node = std::make_shared<tnn::Node>("new_heir_node");
            new_node->info->type = tnn::LAYER_TANH;
            g->nodes.push_back(new_node);
            g->markAllInOneNode(*in);

            return g;
        };

        graph->rewrite(pattern, gen);
    }


    {
        std::vector<std::string> text_graph_pattern = {
            "Add",
            "Mul    Mul<*",
        };

        auto gen = [](std::shared_ptr<tnn::AnchorGraph> in) -> std::shared_ptr<tnn::HeirGraph> {
            if (in->inputs().size() != 1 || in->outputs().size() != 1 ){
                return nullptr;
            }

            auto g = std::make_shared<tnn::HeirGraph>();
            int num_inputs = 1;
            for(int i=0;i<num_inputs;i++) {
                auto ph = g->getNodeByBlobName(std::string("PlaceHolder_") + std::to_string(i));
                ph->info->type = tnn::LAYER_NOT_SUPPORT;
            }
            auto new_node = std::make_shared<tnn::Node>("_norm");
            new_node->info->type = tnn::LAYER_LAYER_NORM;
            g->nodes.push_back(new_node);
            g->markAllInOneNode(*in);

            return g;
        };

        if (parser.parseFromString(text_graph_pattern)) {
            pattern = parser.getGraph();
            std::ofstream f("pattern2.tnnproto");
            pattern->dump(f);
            graph->rewrite(pattern, gen);
        }
    }


    {
        std::vector<std::string> text_graph_pattern = {
            "PlaceHolder PlaceHolder",
            "Mul",
            "Mul+>",
        };

        auto gen = [](std::shared_ptr<tnn::AnchorGraph> in) -> std::shared_ptr<tnn::HeirGraph> {
            if (in->inputs().size() != 2 || in->outputs().size() != 1 ){
                return nullptr;
            }

            auto g = std::make_shared<tnn::HeirGraph>();
            int num_inputs = 2;
            for(int i=0;i<num_inputs;i++) {
                auto ph = g->getNodeByBlobName(std::string("PlaceHolder_") + std::to_string(i));
                ph->info->type = tnn::LAYER_NOT_SUPPORT;
            }
            auto new_node = std::make_shared<tnn::Node>("_mulmul");
            new_node->info->type = tnn::LAYER_CONVOLUTION;
            g->nodes.push_back(new_node);
            g->markAllInOneNode(*in);

            return g;
        };

        if (parser.parseFromString(text_graph_pattern)) {
            pattern = parser.getGraph();
            std::ofstream f("pattern3.tnnproto");
            pattern->dump(f);
            graph->rewrite(pattern, gen);
        }
    }

    std::ofstream f("rewrited.tnnproto");
    graph->dump(f);

    std::string s = tnn::Logger::instance().str();
    // printf("---------final ------------------len%lu:\n%s\n", s.length(), s.c_str());

    return 0;
}
