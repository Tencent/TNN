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

#include "tnn/optimizer/net_optimizer_fuse_conv_activation.h"

#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/text_graph_parser.h"
#include "tnn/optimizer/graph_matcher/graph_matcher.h"
#include "tnn/optimizer/graph_matcher/logger.h"

namespace TNN_NS {

namespace optimizer {

    NetOptimizerRegister<NetOptimizerFuseConvActivation> g_net_optimizer_fuse_conv_act(OptPriority::P1);

    std::string NetOptimizerFuseConvActivation::Strategy() {
        return kNetOptimizerFuseConvActivation;
    }

    bool NetOptimizerFuseConvActivation::IsSupported(const NetworkConfig &net_config) {
        auto device = net_config.device_type;
        if (device == DEVICE_METAL || device == DEVICE_OPENCL || device == DEVICE_ARM || device == DEVICE_NAIVE) {
            kLayerActivationMap[LAYER_RELU]    = ActivationType_ReLU;
            kLayerActivationMap[LAYER_RELU6]   = ActivationType_ReLU6;
            kLayerActivationMap[LAYER_SIGMOID] = ActivationType_SIGMOID_MUL;
            kLayerActivationMap[LAYER_SWISH]   = ActivationType_SIGMOID_MUL;
            return true;
        }
        if (device == DEVICE_RK_NPU) {
            kLayerActivationMap[LAYER_RELU] = ActivationType_ReLU;
            return true;
        }
        if (device == DEVICE_X86 && net_config.network_type != NETWORK_TYPE_OPENVINO) {
            kLayerActivationMap[LAYER_RELU]  = ActivationType_ReLU;
            kLayerActivationMap[LAYER_RELU6] = ActivationType_ReLU6;
            return true;
        }
        return false;
    }

    Status NetOptimizerFuseConvActivation::Optimize(NetStructure *structure, NetResource *resource) {
        // This optimizer is only for illustrating purpose, not running it.
        return TNN_OK;

        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::shared_ptr<Graph> graph = std::make_shared<Graph>();
        RETURN_ON_FAIL(graph->fromInterpreted(structure, resource));

        std::vector<std::string> text_graph_pattern = {
            "Convolution@conv",
            "AnyType@act",
        };

        TextGraphParser parser;
        std::shared_ptr<Graph> pattern = nullptr;
        if (parser.parseFromString(text_graph_pattern)) {
            pattern = parser.getGraph();
        } else {
            return Status(TNNERR_PARAM_ERR, "invalid pattern syntax.");
        }

        auto gen = [&](std::shared_ptr<AnchorGraph> in) -> std::shared_ptr<Graph> {
            if (in->inputs().size() != 1 || in->outputs().size() != 1 ){
                return nullptr;
            }

            auto conv_node = in->getNodeByTensorName(std::string("@conv"));
            auto act_node = in->getNodeByTensorName(std::string("@act"));
            if (!conv_node || ! act_node) {
                WARN("node of interest not found in conv_activation optimizer");
                return nullptr;
            }

            if (kLayerActivationMap.count(act_node->info->type) == 0)  {
                return nullptr;
            }
            INFO("found pattern at Node:%s", conv_node->name().c_str());
            // create new node. 
            // inorder the maintain the net_resoruce map un changed. we create a Node of the same name as before
            auto g = std::make_shared<Graph>();
            auto in_name = "input_1";
            auto in1 = g->getNodeOrCreatePlaceHolder(in_name);
            auto status = g->createNode(LAYER_CONVOLUTION, {in_name}, {conv_node->name()});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_node = g->getNodeByTensorName(conv_node->name());
            new_node->info->param = conv_node->info->param->Copy();
            auto conv_param = dynamic_cast<ConvLayerParam *>(new_node->info->param.get());
            if (!conv_param) {
                return nullptr;
            }

            // update layer param. 
            auto activation_type = kLayerActivationMap[act_node->info->type];
            if (conv_param->quantized)  {
                // quantized conv fuse relu and relu6
                if (activation_type == ActivationType_ReLU || activation_type == ActivationType_ReLU6) {
                    conv_param->activation_type = activation_type;
                } else {
                    return nullptr;
                }
            } else {
                conv_param->activation_type = activation_type;
            }

            return g;
        };

        RETURN_ON_FAIL(graph->rewrite(pattern, gen));

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
