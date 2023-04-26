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

#include "tnn/optimizer/net_optimizer_fuse_layer_norm.h"

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

    NetOptimizerRegister<NetOptimizerFuseLayerNorm> g_net_optimizer_fuse_layer_norm(OptPriority::P0);

    std::string NetOptimizerFuseLayerNorm::Strategy() {
        return kNetOptimizerFuseLayerNorm;
    }

    bool NetOptimizerFuseLayerNorm::IsSupported(const NetworkConfig &net_config) {
        return true;
    }

    Status NetOptimizerFuseLayerNorm::Optimize(NetStructure *structure, NetResource *resource) {

        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::shared_ptr<Graph> graph = std::make_shared<Graph>();
        RETURN_ON_FAIL(graph->fromInterpreted(structure, resource));

        std::string pattern_str = R"(
            graph(%x):
                %reduce_0     = ReduceMean(%x)
                %sub          = Sub(%x, %reduce_0)
                %mul_0        = Mul(%sub, %sub)
                %reduce_1     = ReduceMean(%mul_0)
                %eps          = Add(%reduce_1)
                %sqrt         = Sqrt(%eps)
                %div          = Div(%sub, %sqrt)
                %scale        = Mul(%div)
                %bias         = Add(%scale)
                return (%bias)
        )";

        GraphParser parser;
        std::shared_ptr<Graph> pattern = nullptr;
        if (parser.parseFromString(pattern_str)) {
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

            auto input_node = in->getNodeByTensorName(std::string("@reduce_0"));
            auto scale_node = in->getNodeByTensorName(std::string("@scale"));
            auto bias_node = in->getNodeByTensorName(std::string("@bias"));
            auto eps_node = in->getNodeByTensorName(std::string("@eps"));
            if (!bias_node|| ! scale_node || !input_node || !eps_node) {
                WARN("node of interest not found in layer_norm optimizer");
                return nullptr;
            }

            // TODO, check reduce mean param is: keep_dims=1, axis=-1
            DEBUG("found layernorm pattern at Node:%s", bias_node->name().c_str());

            auto scale_layer_res = dynamic_cast<EltwiseLayerResource *>(resource->resource_map[scale_node->name()].get());
            auto bias_layer_res  = dynamic_cast<EltwiseLayerResource *>(resource->resource_map[bias_node->name()].get());
            auto eps_layer_res   = dynamic_cast<EltwiseLayerResource *>(resource->resource_map[eps_node->name()].get());
            if (!scale_layer_res || !bias_layer_res || !eps_layer_res) {
                ERRORV("Layernorm optimizer got nil resource.", msg);
                return nullptr;
            }

            // create new nodes. 
            auto g = std::make_shared<Graph>();

            std::string in_name = input_node->info->inputs[0];
            std::string scale_name = in_name + "_layernorm_scale_";
            std::string bias_name  = in_name + "_layernorm_bias_";
            std::string output_name = in_name + "_layernorm_output_";

            auto in1 = g->getNodeOrCreatePlaceHolder(in_name);
            RETURN_VALUE_ON_NEQ(g->createConst(scale_name, std::make_shared<RawBuffer>(scale_layer_res->element_handle)), TNN_OK, nullptr);
            RETURN_VALUE_ON_NEQ(g->createConst(bias_name,  std::make_shared<RawBuffer>( bias_layer_res->element_handle)), TNN_OK, nullptr);

            CREATE_NODE(new_node, g, LAYER_LAYER_NORM, NAMES({in_name, scale_name, bias_name}), {output_name});

            RETURN_VALUE_ON_NEQ(new_node->createParam<LayerNormLayerParam>(), TNN_OK, nullptr);
            std::shared_ptr<float> eps_ptr = GetFloatFromRawBuffer(eps_layer_res->element_handle);
            new_node->param<LayerNormLayerParam>()->eps = *eps_ptr;
            new_node->param<LayerNormLayerParam>()->reduce_dims_size = 1;

            return g;
        };

        RETURN_ON_FAIL(graph->rewrite(pattern, gen));

        // ModelPacker packer(structure, resource);
        // packer.Pack("pack.tnnproto", "pack.tnnmodel");

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS

