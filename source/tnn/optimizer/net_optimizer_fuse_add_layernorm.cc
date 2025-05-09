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

#include "tnn/optimizer/net_optimizer_fuse_add_layernorm.h"

#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/graph_parser.h"
#include "tnn/optimizer/graph_matcher/graph_matcher.h"
#include "tnn/optimizer/graph_matcher/logger.h"
// #include "tnn/interpreter/tnn/model_packer.h"

namespace TNN_NS {

namespace optimizer {

    NetOptimizerRegister<NetOptimizerFuseAddLayerNorm> g_net_optimizer_fuse_add_layernorm(OptPriority::P1);

    std::string NetOptimizerFuseAddLayerNorm::Strategy() {
        return kNetOptimizerFuseAddLayerNorm;
    }

    bool NetOptimizerFuseAddLayerNorm::IsSupported(const NetworkConfig &net_config) {
        if (net_config.precision == PRECISION_HIGH) {
            return false;
        }

        auto device = net_config.device_type;
        if (device == DEVICE_CUDA) {
            return true;
        }
        return false;
    }

    struct LNPattenInfo {
        std::string graph_str;
        std::string bias_node_name;
        std::string ln_node_name;
    };

    class LNRewriter {
        public:
            LNRewriter(std::shared_ptr<Graph> graph, NetStructure *structure, NetResource *resource)
                : graph_(graph), structure_(structure), resource_(resource) {
            }

            Status Rewrite(const LNPattenInfo &patten_info);

        private:
            std::shared_ptr<Graph> graph_;
            NetStructure *structure_;
            NetResource *resource_;
    };

    Status NetOptimizerFuseAddLayerNorm::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }
        // TNN_NS::Logger::instance().set_verbose_level("D");

        std::shared_ptr<Graph> graph = std::make_shared<Graph>();
        RETURN_ON_FAIL(graph->fromInterpreted(structure, resource));

        LNRewriter rewriter(graph, structure, resource);

        for (const auto &patten : GetLNPattens()) {
            RETURN_ON_FAIL(rewriter.Rewrite(patten));
        }

        // TNN_NS::Logger::instance().set_verbose_level("W");

        return TNN_OK;
    }

    std::vector<LNPattenInfo> NetOptimizerFuseAddLayerNorm::GetLNPattens() {
        std::vector<LNPattenInfo> pattens;

        {
            LNPattenInfo ln_patten;
            ln_patten.graph_str = R"(
                graph(%att_out, %res_in, %scale, %bias):
                    %bias_out     = Add(%att_out)
                    %residual_out = Add(%bias_out, %res_in)
                    %out          = LayerNorm(%residual_out, %scale, %bias)
                    return (%out)
            )";
            ln_patten.bias_node_name = "@bias_out";
            ln_patten.ln_node_name   = "@out";
            pattens.push_back(ln_patten);
        }

        {
            LNPattenInfo ln_patten;
            ln_patten.graph_str = R"(
                graph(%att_out, %res_in, %scale, %bias):
                    %bias_out     = Add(%att_out)
                    %residual_out = Add(%res_in, %bias_out)
                    %out          = LayerNorm(%residual_out, %scale, %bias)
                    return (%out)
            )";
            ln_patten.bias_node_name = "@bias_out";
            ln_patten.ln_node_name   = "@out";
            pattens.push_back(ln_patten);
        }

        return pattens;
    }

    Status LNRewriter::Rewrite(const LNPattenInfo &patten_info) {
        GraphParser parser;
        std::shared_ptr<Graph> pattern = nullptr;
        if (parser.parseFromString(patten_info.graph_str)) {
            pattern = parser.getGraph();
        } else {
            return Status(TNNERR_PARAM_ERR, "invalid pattern syntax.");
        }

        auto gen = [&](std::shared_ptr<AnchorGraph> in) -> std::shared_ptr<Graph> {
            if (in->inputs().size() != 4 || in->outputs().size() != 1 ) {
                return nullptr;
            }

            auto bias_add_node = in->getNodeByTensorName(patten_info.bias_node_name);
            if (!bias_add_node) {
                WARN("node of interest not found");
                return nullptr;
            }

            auto layernorm_node = in->getNodeByTensorName(patten_info.ln_node_name);
            if (!layernorm_node) {
                WARN("node of interest not found");
                return nullptr;
            }

            auto layer_norm_param = dynamic_cast<LayerNormLayerParam *>(layernorm_node->info->param.get());
            if (!layer_norm_param) {
                WARN("layer_norm_param is nil");
                return nullptr;
            }

            auto g = std::make_shared<Graph>();
            auto in_att_out_name = "input_att_out";
            auto in1 = g->getNodeOrCreatePlaceHolder(in_att_out_name);
            auto in_res_in_name = "input_res_in";
            auto in2 = g->getNodeOrCreatePlaceHolder(in_res_in_name);
            auto in_scale_name = "input_scale";
            auto in3 = g->getNodeOrCreatePlaceHolder(in_scale_name);
            auto in_bias_name = "input_bias";
            auto in4 = g->getNodeOrCreatePlaceHolder(in_bias_name);

            auto out_name = bias_add_node->name();
            CREATE_NODE(fused_node, g, LAYER_FUSED, NAMES({in_att_out_name, in_res_in_name, in_scale_name, in_bias_name}), {out_name});

            RETURN_VALUE_ON_NEQ(fused_node->createParam<FusedLayerParam>(), TNN_OK, nullptr);
            fused_node->param<FusedLayerParam>()->type = FusionType_AddBiasResidualLayerNorm;
            fused_node->param<FusedLayerParam>()->layer_norm_param = *layer_norm_param;

            return g;
        };

        RETURN_ON_FAIL(graph_->rewrite(pattern, gen));

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
