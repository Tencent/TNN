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

#include "tnn/optimizer/net_optimizer_fuse_split_gelu.h"

#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/interpreter/tnn/model_packer.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/graph_parser.h"
#include "tnn/optimizer/graph_matcher/graph_matcher.h"
#include "tnn/optimizer/graph_matcher/logger.h"

namespace TNN_NS {

namespace optimizer {

    NetOptimizerRegister<NetOptimizerFuseSplitGELU> g_net_optimizer_fuse_split_gelu(OptPriority::P0);

    std::string NetOptimizerFuseSplitGELU::Strategy() {
        return kNetOptimizerFuseSplitGELU;
    }

    bool NetOptimizerFuseSplitGELU::IsSupported(const NetworkConfig &net_config) {
        return true;
    }

    Status NetOptimizerFuseSplitGELU::Optimize(NetStructure *structure, NetResource *resource) {

        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::shared_ptr<Graph> graph = std::make_shared<Graph>();
        RETURN_ON_FAIL(graph->fromInterpreted(structure, resource));

        std::string pattern_str = R"(
            graph(%in):
                %state, %gate      = SplitV(%in)
                %gelu              = GELU(%gate)
                %out               = Mul(%state, %gelu)
                return (%out)
        )";

        GraphParser parser;
        std::shared_ptr<Graph> pattern = nullptr;
        if (parser.parseFromString(pattern_str)) {
            pattern = parser.getGraph();
        } else {
            LOGEV("%s", msg, "invalid pattern syntax.");
            return Status(TNNERR_PARAM_ERR, msg);
        }

        auto gen = [&](std::shared_ptr<AnchorGraph> in) -> std::shared_ptr<Graph> {
            if (in->inputs().size() != 1 || in->outputs().size() != 1 ){
                return nullptr;
            }

            // create new nodes. 
            auto g = std::make_shared<Graph>();
            std::vector<std::string> in_names = {in->inputNodes()[0]->name()};
            std::vector<std::string> out_names = {in->outputNodes()[0]->name()};
            g->getNodeOrCreatePlaceHolder(in_names[0]);
            auto status = g->createNode(LAYER_FUSED_SPLIT_GELU, in_names, out_names);
            if (status != TNN_OK) {
                return nullptr;
            }
            auto split_gelu_node         = g->getNodeByTensorName(out_names[0]);
            split_gelu_node->createParam<LayerParam>();
            
            return g;
        };

        RETURN_ON_FAIL(graph->rewrite(pattern, gen));

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS

