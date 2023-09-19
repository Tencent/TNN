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

#include "tnn/optimizer/net_optimizer_fuse_ffn.h"

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

    NetOptimizerRegister<NetOptimizerFuseFFN> g_net_optimizer_fuse_ffn(OptPriority::P1);

    std::string NetOptimizerFuseFFN::Strategy() {
        return kNetOptimizerFuseFFN;
    }

    bool NetOptimizerFuseFFN::IsSupported(const NetworkConfig &net_config) {
        // May lead to potential bugs, Closed Right now.
        return false;

        if (net_config.precision == PRECISION_HIGH) {
            return false;
        }

        auto device = net_config.device_type;
        if (device == DEVICE_CUDA) {
            return true;
        }
        return false;
    }

    struct FFNPattenInfo {
        std::string graph_str;
        std::string mmin_node_name;
        std::string add_node_name;
        std::string mmout_node_name;
    };

    class FFNRewriter {
        public:
            FFNRewriter(std::shared_ptr<Graph> graph, NetStructure *structure, NetResource *resource)
                : graph_(graph), structure_(structure), resource_(resource) {
            }

            Status Rewrite(const FFNPattenInfo &patten_info);

        private:
            std::shared_ptr<Graph> graph_;
            NetStructure *structure_;
            NetResource *resource_;
    };

    Status NetOptimizerFuseFFN::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }
        // TNN_NS::Logger::instance().set_verbose_level("D");

        std::shared_ptr<Graph> graph = std::make_shared<Graph>();
        RETURN_ON_FAIL(graph->fromInterpreted(structure, resource));

        FFNRewriter rewriter(graph, structure, resource);

        for (const auto &patten : GetFFNPattens()) {
            RETURN_ON_FAIL(rewriter.Rewrite(patten));
        }

        // TNN_NS::Logger::instance().set_verbose_level("W");

        return TNN_OK;
    }

    std::vector<FFNPattenInfo> NetOptimizerFuseFFN::GetFFNPattens() {
        std::vector<FFNPattenInfo> pattens;

        // GELU
        {
            FFNPattenInfo gelu_patten;
            gelu_patten.graph_str = R"(
                graph(%in):
                    %mid_out = MatMul(%in)
                    %add_out = Add(%mid_out)
                    %act_out = GELU(%add_out)
                    %out     = MatMul(%act_out)
                    return (%out)
            )";
            gelu_patten.mmin_node_name = "@mid_out";
            gelu_patten.add_node_name = "@add_out";
            gelu_patten.mmout_node_name = "@out";
            pattens.push_back(gelu_patten);
        }

        // GELU_new
        {
            // 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
            FFNPattenInfo new_gelu_patten;
            new_gelu_patten.graph_str = R"(
                graph(%in):
                    %mid_out  = MatMul(%in)
                    %add_out  = Add(%mid_out)
                    %mul_1    = Mul(%add_out)
                    %pow_out  = Power(%add_out)
                    %mul_2    = Mul(%pow_out)
                    %add_1    = Add(%add_out, %mul_2)
                    %mul_3    = Mul(%add_1)
                    %tanh_out = Tanh(%mul_3)
                    %add_2    = Add(%tanh_out)
                    %mul_4    = Mul(%mul_1, %add_2)
                    %out      = MatMul(%mul_4)
                    return (%out)
            )";
            new_gelu_patten.mmin_node_name = "@mid_out";
            new_gelu_patten.add_node_name = "@add_out";
            new_gelu_patten.mmout_node_name = "@out";
            pattens.push_back(new_gelu_patten);
        }

        // GELU_Fast
        {
            // 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * input * (1.0 + 0.044715 * input * input)))
            FFNPattenInfo fast_gelu_patten;
            fast_gelu_patten.graph_str = R"(
                graph(%in):
                    %mid_out  = MatMul(%in)
                    %add_out  = Add(%mid_out)
                    %mul_1    = Mul(%add_out)
                    %mul_2    = Mul(%add_out)
                    %mul_3    = Mul(%add_out)
                    %mul_4    = Mul(%mul_3, %add_out)
                    %add_1    = Add(%mul_4)
                    %mul_5    = Mul( %mul_2, %add_1)
                    %tanh_out = Tanh(%mul_5)
                    %add_2    = Add(%tanh_out)
                    %mul_6    = Mul(%mul_1, %add_2)
                    %out      = MatMul(%mul_6)
                    return (%out)
            )";
            fast_gelu_patten.mmin_node_name = "@mid_out";
            fast_gelu_patten.add_node_name = "@add_out";
            fast_gelu_patten.mmout_node_name = "@out";
            pattens.push_back(fast_gelu_patten);
        }

        return pattens;
    }

    Status FFNRewriter::Rewrite(const FFNPattenInfo &patten_info) {
        GraphParser parser;
        std::shared_ptr<Graph> pattern = nullptr;
        if (parser.parseFromString(patten_info.graph_str)) {
            pattern = parser.getGraph();
        } else {
            return Status(TNNERR_PARAM_ERR, "invalid pattern syntax.");
        }

        auto gen = [&](std::shared_ptr<AnchorGraph> in) -> std::shared_ptr<Graph> {
            if (in->inputs().size() != 1 || in->outputs().size() != 1 ) {
                return nullptr;
            }

            // MatMul
            auto matmul_in_node = in->getNodeByTensorName(std::string(patten_info.mmin_node_name));
            if (!matmul_in_node) {
                WARN("node of interest not found");
                return nullptr;
            }
            auto matmul_in_param = dynamic_cast<MatMulLayerParam *>(matmul_in_node->info->param.get());
            if (!matmul_in_param) {
                WARN("matmul_in_param is nil");
                return nullptr;
            }
            if (matmul_in_param->weight_position != 1) {
                WARN("matmul_in_param weight_position not supported");
                return nullptr;
            }
            if (resource_->resource_map.find(matmul_in_node->info->name) == resource_->resource_map.end()) {
                WARN("matmul_in_resource is not found");
                return nullptr;
            }
            MatMulLayerResource *ffn_matmul_in = dynamic_cast<MatMulLayerResource *>(resource_->resource_map[matmul_in_node->info->name].get());
            if (!ffn_matmul_in) {
                WARN("matmul_in_resource is nil");
                return nullptr;
            }
            auto weight_dims = ffn_matmul_in->weight.GetBufferDims();
            if (weight_dims.size() != 2) {
                WARN("matmul_in_resource dims not support");
                return nullptr;
            }
            int inter_size = weight_dims[weight_dims.size() - 1];

            // Add
            auto bias_add_node = in->getNodeByTensorName(std::string(patten_info.add_node_name));
            if (!bias_add_node) {
                WARN("node of interest not found");
                return nullptr;
            }
            if (resource_->resource_map.find(bias_add_node->info->name) == resource_->resource_map.end()) {
                WARN("bias_add_resource is not found");
                return nullptr;
            }
            EltwiseLayerResource *ffn_bias = dynamic_cast<EltwiseLayerResource *>(resource_->resource_map[bias_add_node->info->name].get());
            if (!ffn_bias) {
                WARN("bias_add_resource is nil");
                return nullptr;
            }

            // MatMul
            auto matmul_out_node = in->getNodeByTensorName(std::string(patten_info.mmout_node_name));
            if (!matmul_out_node) {
                WARN("node of interest not found");
                return nullptr;
            }
            auto matmul_out_param = dynamic_cast<MatMulLayerParam *>(matmul_out_node->info->param.get());
            if (!matmul_out_param) {
                WARN("matmul_out_param is nil");
                return nullptr;
            }
            if (matmul_out_param->weight_position != 1) {
                WARN("matmul_out_param weight_position not supported");
                return nullptr;
            }
            if (resource_->resource_map.find(matmul_out_node->info->name) == resource_->resource_map.end()) {
                WARN("matmul_out_resource is not found");
                return nullptr;
            }
            MatMulLayerResource *ffn_matmul_out = dynamic_cast<MatMulLayerResource *>(resource_->resource_map[matmul_out_node->info->name].get());
            if (!ffn_matmul_out) {
                WARN("matmul_out_resource is nil");
                return nullptr;
            }
            weight_dims = ffn_matmul_out->weight.GetBufferDims();
            if (weight_dims.size() != 2) {
                WARN("matmul_out_resource dims not support");
                return nullptr;
            }
            if (inter_size != weight_dims[0]) {
                WARN("matmul_out_resource inter_size not match");
                return nullptr;
            }

            auto g = std::make_shared<Graph>();
            auto in_name = "input";
            auto in1 = g->getNodeOrCreatePlaceHolder(in_name);

            auto out_name = matmul_out_node->name() + "__ffn__";
            auto status = g->createNode(LAYER_FUSED, {in_name}, {out_name});
            if (status != TNN_OK) {
                return nullptr;
            }

            auto fused_node = g->getNodeByTensorName(out_name);

            auto fused_param = std::make_shared<FusedLayerParam>();
            fused_param->type = FusionType_FFN;
            fused_param->ffn_activation = ActivationType_GELU;
            fused_param->ffn_inter_size = inter_size;
            fused_node->info->param = fused_param;

            auto fused_resource = std::make_shared<FusedLayerResource>();
            if (resource_->resource_map.find(out_name) != resource_->resource_map.end()) {
                WARN("fused_resource name conflict");
                return nullptr;
            }
            fused_resource->ffn_matmul_in = *ffn_matmul_in;
            fused_resource->ffn_matmul_out = *ffn_matmul_out;
            fused_resource->ffn_bias = *ffn_bias;
            resource_->resource_map[out_name] = fused_resource;

            return g;
        };

        RETURN_ON_FAIL(graph_->rewrite(pattern, gen));

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
