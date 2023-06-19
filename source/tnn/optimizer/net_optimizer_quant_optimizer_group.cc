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

#include "tnn/optimizer/net_optimizer_quant_optimizer_group.h"

#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/text_graph_parser.h"
#include "tnn/optimizer/graph_matcher/graph_matcher.h"
#include "tnn/optimizer/graph_matcher/logger.h"

namespace TNN_NS {

namespace optimizer {

    NetOptimizerRegister<NetOptimizerQuantOptimizerGroup> g_net_optimizer_quant_optimizer_group(OptPriority::P2);

    std::string NetOptimizerQuantOptimizerGroup::Strategy() {
        return kNetOptimizerQuantOptimizerGroup;
    }

    bool NetOptimizerQuantOptimizerGroup::IsSupported(const NetworkConfig &net_config) {
        auto device = net_config.device_type;
        if (device == DEVICE_CUDA) {
            return true;
        }
        return false;
    }

    Status NetOptimizerQuantOptimizerGroup::Optimize(NetStructure *structure, NetResource *resource) {

        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::shared_ptr<Graph> graph = std::make_shared<Graph>();
        RETURN_ON_FAIL(graph->fromInterpreted(structure, resource));

        std::vector<std::string> text_graph_pattern = {
            "AnyType@x_q                 ",
            "Dequantize@dq  Permute@rhs",
            "MatMul+>@fc",
            "Add@bias",
            "Quantize@q",
        };

        TextGraphParser parser;
        std::shared_ptr<Graph> pattern = nullptr;
        if (parser.parseFromString(text_graph_pattern)) {
            pattern = parser.getGraph();
        } else {
            LOGEV("%s", msg, "invalid pattern syntax.");
            return Status(TNNERR_PARAM_ERR, msg);
        }

        // Logger::instance().set_verbose_level("I");

        auto gen = [&](std::shared_ptr<AnchorGraph> in) -> std::shared_ptr<Graph> {

            if (in->inputs().size() != 1 || in->outputs().size() != 1 ){
                return nullptr;
            }

            auto bias_node = in->getNodeByTensorName(std::string("@bias"));
            auto q_node = in->getNodeByTensorName(std::string("@q"));
            if (!bias_node|| ! q_node) {
                WARN("node of interest not found in quanti optimizer");
                return nullptr;
            }


            // printf("Got node add of name %s\n", bias_node->name().c_str());
            // INFO("found pattern at Node:%s", bias_node->name().c_str());

            auto add_layer_res = dynamic_cast<EltwiseLayerResource *>(resource->resource_map[bias_node->name()].get());
            if (!add_layer_res) {
                ERRORV("bias node of name %s, got nil resource.", msg, bias_node->name().c_str());
                return nullptr;
            }
            auto q_layer_res = dynamic_cast<QuantizeLayerResource *>(resource->resource_map[q_node->name()].get());
            if (!q_layer_res) {
                ERRORV("q node of name %s, got nil resource.", msg, q_node->name().c_str());
                return nullptr;
            }

            // printf("bias node handle len : %lu\n", add_layer_res->element_handle.GetDataCount());
            // printf("q    node handle len : %lu\n", q_layer_res->scale_handle.GetDataCount());

            return nullptr;
            // create new node. 
            // inorder the maintain the net_resoruce map un changed. we create a Node of the same name as before
            auto g = std::make_shared<Graph>();
            auto in_name = "input_1";
            auto in1 = g->getNodeOrCreatePlaceHolder(in_name);
            auto status = g->createNode(LAYER_CONVOLUTION, {in_name}, {bias_node->name()});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_node = g->getNodeByTensorName(bias_node->name());
            new_node->info->param = bias_node->info->param->Copy();
            auto conv_param = dynamic_cast<ConvLayerParam *>(new_node->info->param.get());
            if (!conv_param) {
                return nullptr;
            }

            // // update layer param. 
            // auto activation_type = kLayerActivationMap[act_node->info->type];
            // if (conv_param->quantized)  {
            //     // quantized conv fuse relu and relu6
            //     if (activation_type == ActivationType_ReLU || activation_type == ActivationType_ReLU6) {
            //         conv_param->activation_type = activation_type;
            //     } else {
            //         return nullptr;
            //     }
            // } else {
            //     conv_param->activation_type = activation_type;
            // }

            // printf("finish gen \n");
            return g;
        };

        RETURN_ON_FAIL(graph->rewrite(pattern, gen));

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS

