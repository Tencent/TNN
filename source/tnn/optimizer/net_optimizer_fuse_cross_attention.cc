// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "tnn/optimizer/net_optimizer_fuse_cross_attention.h"

#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/graph_matcher.h"
#include "tnn/optimizer/graph_matcher/graph_parser.h"
#include "tnn/optimizer/graph_matcher/logger.h"

namespace TNN_NS {

namespace optimizer {

    NetOptimizerRegister<NetOptimizerFuseCrossAttention> g_net_optimizer_fuse_cross_attention(OptPriority::P1);

    std::string NetOptimizerFuseCrossAttention::Strategy() {
        return kNetOptimizerFuseCrossAttention;
    }

    bool NetOptimizerFuseCrossAttention::IsSupported(const NetworkConfig &net_config) {
        return false;
        if (net_config.precision == PRECISION_HIGH) {
            return false;
        }
        auto device = net_config.device_type;
        if (device == DEVICE_CUDA) {
            // TODO: only support several sm version
            return true;
        }
        return false;
    }

    struct CrossAttentionPatternInfo {
        std::string graph_str;
        int nb_inputs;
        int nb_outputs;
        std::string shape_node_name;
        std::string div_node_name;
        std::string output_node_name;
        int cast_mask_ = -1;
        bool has_attention_mask = true;
    };

    class CrossAttentionRewriter {
        public:
            CrossAttentionRewriter(std::shared_ptr<Graph> graph, NetStructure *structure, NetResource *resource)
                : graph_(graph), structure_(structure), resource_(resource) {
            }

            Status Rewrite(const CrossAttentionPatternInfo &patten_info);

        private:
            MatMulLayerResource *GetWeight(std::shared_ptr<AnchorGraph> in, const std::string &mm_node_name);
            std::vector<std::string> GetInputs(std::shared_ptr<Graph> g, const CrossAttentionPatternInfo &info);
            std::vector<std::string> GetOutputs(std::shared_ptr<AnchorGraph> in, const CrossAttentionPatternInfo &info);

            std::shared_ptr<Graph> graph_;
            NetStructure *structure_;
            NetResource *resource_;

            int head_num_;
            int size_per_head_;
            int hidden_size_;
            float q_scaling_;
    };

    Status NetOptimizerFuseCrossAttention::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }
        if (!resource) {
            LOGE("Error: empty NetResource\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetResource");
        }
        // TNN_NS::Logger::instance().set_verbose_level("D");

        std::shared_ptr<Graph> graph = std::make_shared<Graph>();
        RETURN_ON_FAIL(graph->fromInterpreted(structure, resource));

        CrossAttentionRewriter rewriter(graph, structure, resource);

        for (const auto &patten : GetAttentionPattens()) {
            RETURN_ON_FAIL(rewriter.Rewrite(patten));
        }

        return TNN_OK;
    }

    Status NetOptimizerFuseCrossAttention::EliminateRedundantCasts(NetStructure *structure, NetResource *resource) {
        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();

        std::unordered_map<std::string, std::string> cast_map;
        std::vector<std::shared_ptr<LayerInfo>> layers_optimized;
        for (int index = 0; index < count; index++) {
            auto layer_info_curr = layers_orig[index];

            if (layer_info_curr->type != LAYER_CAST || layer_info_curr->inputs.size() != 1 || layer_info_curr->outputs.size() != 1 ||
                structure->outputs.find(layer_info_curr->outputs[0]) != structure->outputs.end()) {
                layers_optimized.push_back(layers_orig[index]);
                continue;
            }

            auto curr_param = dynamic_cast<CastLayerParam *>(layer_info_curr->param.get());
            if (!curr_param) {
                continue;
            }

            std::string key = layer_info_curr->inputs[0] + "_cast_to_" + std::to_string(curr_param->to);
            if (cast_map.find(key) == cast_map.end()) {
                cast_map[key] = layer_info_curr->outputs[0];
                layers_optimized.push_back(layers_orig[index]);
            } else {
                for (int j = index; j < count; ++j) {
                    auto layer_info_after = layers_orig[j];
                    for (int i = 0; i < layer_info_after->inputs.size(); ++i) {
                        if (layer_info_after->inputs[i] == layer_info_curr->outputs[0]) {
                            layer_info_after->inputs[i] = cast_map[key];
                        }
                    }
                }
            }
        }
        structure->layers = layers_optimized;

        return TNN_OK;
    }

    std::vector<CrossAttentionPatternInfo> NetOptimizerFuseCrossAttention::GetAttentionPattens() {
        std::vector<CrossAttentionPatternInfo> pattens;

        // CrossAttention
        {
            CrossAttentionPatternInfo cross_attention_patten;
            cross_attention_patten.graph_str = R"(
                graph(%q_in, kv_in, %num_heads):
                    %q_linear_mm          = MatMul(%q_in)
                    %q_batch_shape        = Shape(%q_linear_mm)
                    %q_batch_gather       = Gather(%q_batch_shape)
                    %q_batch              = Unsqueeze(%q_batch_gather)
                    %q_seqlen_shape        = Shape(%q_linear_mm)
                    %q_seqlen_gather       = Gather(%q_seqlen_shape)
                    %q_seqlen              = Unsqueeze(%q_seqlen_gather)
                    %q_hidden_size_shape       = Shape(%q_linear_mm)
                    %q_hidden_size_gather      = Gather(%q_hidden_size_shape)
                    %q_hidden_size             = Unsqueeze(%q_hidden_size_gather)
                    %q_per_head_size        = Div(%q_hidden_size)
                    %q_reshape_shape      = Concat(%q_batch, %q_seqlen, %num_heads, %q_per_head_size)
                    %q_reshape            = ReshapeTorch(%q_linear_mm, %q_reshape_shape)
                    %q_permute            = Permute(%q_reshape)
                    %q_batch_mul_heads  = Mul(%q_batch)
                    %q_reshape_batch_mul_heads_shape = Concat(%q_batch_mul_heads, %q_seqlen, %q_per_head_size)
                    %q_reshape_batch_mul_heads = ReshapeTorch(%q_permute, %q_reshape_batch_mul_heads_shape)
                    %k_linear_mm          = MatMul(%kv_in)
                    %v_linear_mm          = MatMul(%kv_in)
                    %k_batch_shape        = Shape(%k_linear_mm)
                    %k_batch_gather       = Gather(%k_batch_shape)
                    %k_batch              = Unsqueeze(%k_batch_gather)
                    %k_seqlen_shape        = Shape(%k_linear_mm)
                    %k_seqlen_gather       = Gather(%k_seqlen_shape)
                    %k_seqlen              = Unsqueeze(%k_seqlen_gather) 
                    %k_hidden_size_shape         = Shape(%k_linear_mm)
                    %k_hidden_size_gather        = Gather(%k_hidden_size_shape)
                    %k_hidden_size               = Unsqueeze(%k_hidden_size_gather) 
                    %k_per_head_size             = Div(%k_hidden_size)
                    %k_reshape_shape      = Concat(%k_batch, %k_seqlen, %num_heads, %k_per_head_size)
                    %k_reshape            = ReshapeTorch(%k_linear_mm, %k_reshape_shape)
                    %k_permute            = Permute(%k_reshape)                   
                    %k_batch_mul_heads  = Mul(%k_batch)
                    %k_reshape_batch_mul_heads_shape = Concat(%k_batch_mul_heads, %k_seqlen, %k_per_head_size)
                    %k_reshape_batch_mul_heads = ReshapeTorch(%k_permute, %k_reshape_batch_mul_heads_shape)
                    %v_batch_shape        = Shape(%v_linear_mm)
                    %v_batch_gather       = Gather(%v_batch_shape)
                    %v_batch              = Unsqueeze(%v_batch_gather) 
                    %v_seqlen_shape        = Shape(%v_linear_mm)
                    %v_seqlen_gather       = Gather(%v_seqlen_shape)
                    %v_seqlen              = Unsqueeze(%v_seqlen_gather)
                    %v_hidden_size_shape         = Shape(%v_linear_mm)
                    %v_hidden_size_gather        = Gather(%v_hidden_size_shape)
                    %v_hidden_size               = Unsqueeze(%v_hidden_size_gather) 
                    %v_per_head_size             = Div(%v_hidden_size)
                    %v_reshape_shape      = Concat(%v_batch, %v_seqlen, %num_heads, %v_per_head_size)
                    %v_reshape            = ReshapeTorch(%v_linear_mm, %v_reshape_shape)
                    %v_permute            = Permute(%v_reshape)                   
                    %v_batch_mul_heads  = Mul(%v_batch)
                    %v_reshape_batch_mul_heads_shape = Concat(%v_batch_mul_heads, %v_seqlen, %v_per_head_size)
                    %v_reshape_batch_mul_heads = ReshapeTorch(%v_permute, %v_reshape_batch_mul_heads_shape)
                    %q_remove_batch_shape = Shape(%q_reshape_batch_mul_heads)
                    %q_remove_batch_gather = Gather(%q_remove_batch_shape)
                    %q_remove_batch = Unsqueeze(%q_remove_batch_gather)
                    %q_remove_seqlen_shape = Shape(%q_reshape_batch_mul_heads)
                    %q_remove_seqlen_gather = Gather(%q_remove_seqlen_shape)
                    %q_remove_seqlen = Unsqueeze(%q_remove_seqlen_gather)
                    %k_remove_hidden_size_shape = Shape(%k_reshape_batch_mul_heads)
                    %k_remove_hidden_size_gather = Gather(%k_remove_hidden_size_shape)
                    %k_remove_hidden_size = Unsqueeze(%k_remove_hidden_size_gather)
                    %remove_shape = Concat(%q_remove_batch, %q_remove_seqlen, %k_remove_hidden_size)
                    %k_permute_trans      = PermuteV2(%k_reshape_batch_mul_heads)
                    %attn_score           = MatMul(%q_reshape_batch_mul_heads, %k_permute_trans)
                    %attn_score_mul       = Mul(%attn_score)
                    %attn_score_softmax   = SoftmaxCaffe(%attn_score_mul)
                    %attn_score_softmax_cast = Cast(%attn_score_softmax)
                    %attn_context         = MatMul(%attn_score_softmax_cast, %v_reshape_batch_mul_heads)
                    %ac_batch_mul_heads_shape       = Shape(%attn_context)
                    %ac_batch_mul_heads_gather      = Gather(%ac_batch_mul_heads_shape)
                    %ac_batch_mul_heads             = Unsqueeze(%ac_batch_mul_heads_gather)
                    %ac_seqlen_shape      = Shape(%attn_context)
                    %ac_seqlen_gather     = Gather(%ac_seqlen_shape)
                    %ac_seqlen            = Unsqueeze(%ac_seqlen_gather)
                    %ac_per_head_size_shape       = Shape(%attn_context)
                    %ac_per_head_size_gather      = Gather(%ac_per_head_size_shape)
                    %ac_per_head_size             = Unsqueeze(%ac_per_head_size_gather)
                    %ac_batch                   = Div(%ac_batch_mul_heads)
                    %ac_reshape_shape           = Concat(%ac_batch, %num_heads, %ac_seqlen, %ac_per_head_size)
                    %ac_reshape                 = ReshapeTorch(%attn_context, %ac_reshape_shape)
                    %ac_permute                 = Permute(%ac_reshape)
                    %ac_hidden_size             = Mul(%ac_per_head_size)
                    %ac_permute_reshape_shape = Concat(%ac_batch, %ac_seqlen, %ac_hidden_size)
                    %ac_permute_reshape = ReshapeTorch(%ac_permute, %ac_permute_reshape_shape)
                    %o_linear_mm = MatMul(%ac_permute_reshape)
                    return (%o_linear_mm, %remove_shape) 
            )";

            cross_attention_patten.nb_inputs        = 3;
            cross_attention_patten.nb_outputs       = 2;
            cross_attention_patten.shape_node_name  = "@q_reshape_shape";
            cross_attention_patten.output_node_name = "@o_linear_mm";
            pattens.push_back(cross_attention_patten);
        }

        return pattens;
    }

    Status CrossAttentionRewriter::Rewrite(const CrossAttentionPatternInfo &patten_info) {
        GraphParser parser;
        std::shared_ptr<Graph> pattern = nullptr;
        if (parser.parseFromString(patten_info.graph_str)) {
            pattern = parser.getGraph();
        } else {
            return Status(TNNERR_PARAM_ERR, "invalid pattern syntax.");
        }

        auto gen = [&](std::shared_ptr<AnchorGraph> in) -> std::shared_ptr<Graph> {
            if (in->inputs().size() != patten_info.nb_inputs || in->outputs().size() != patten_info.nb_outputs) {
                return nullptr;
            }

            auto matmul_q = GetWeight(in, "@q_linear_mm");
            auto matmul_k = GetWeight(in, "@k_linear_mm");
            auto matmul_v = GetWeight(in, "@v_linear_mm");
            auto matmul_o = GetWeight(in, "@o_linear_mm");
            if (!matmul_q || !matmul_k || !matmul_v || !matmul_o) {
                WARN("matmul resource is nil");
                return nullptr;
            }

            auto g         = std::make_shared<Graph>();
            auto in_names  = GetInputs(g, patten_info);
            auto out_names = GetOutputs(in, patten_info);

            auto status = g->createNode(LAYER_FUSED, in_names, out_names);
            if (status != TNN_OK) {
                return nullptr;
            }

            auto fused_node = g->getNodeByTensorName(out_names[0]);
            auto fused_param                     = std::make_shared<FusedLayerParam>();
            fused_param->type                    = FusionType_Attention;
            fused_node->info->param              = fused_param;

            return g;
        };

        RETURN_ON_FAIL(graph_->rewrite(pattern, gen));

        return TNN_OK;
    }

    MatMulLayerResource *CrossAttentionRewriter::GetWeight(std::shared_ptr<AnchorGraph> in, const std::string &mm_node_name) {
        auto matmul_node = in->getNodeByTensorName(mm_node_name);
        if (!matmul_node) {
            WARN("node of interest not found");
            return nullptr;
        }
        auto matmul_param = dynamic_cast<MatMulLayerParam *>(matmul_node->info->param.get());
        if (!matmul_param) {
            WARN("matmul_param is nil");
            return nullptr;
        }
        if (matmul_param->weight_position != 1) {
            WARN("matmul_param weight_position not supported");
            return nullptr;
        }
        auto node_info_name = matmul_node->info->name;
        if (resource_->resource_map.find(node_info_name) == resource_->resource_map.end()) {
            WARN("matmul_resource is not found");
            return nullptr;
        }
        auto matmul_res = dynamic_cast<MatMulLayerResource *>(resource_->resource_map[node_info_name].get());
        if (!matmul_res) {
            WARN("matmul_resource is nil");
            return nullptr;
        }
        auto matmul_weight_dims = matmul_res->weight.GetBufferDims();
        if (matmul_weight_dims.size() != 2) {
            WARN("matmul_resource dims not support");
            return nullptr;
        }
        return matmul_res;
    }

    std::vector<std::string> CrossAttentionRewriter::GetInputs(std::shared_ptr<Graph> g, const CrossAttentionPatternInfo &info) {
        std::vector<std::string> inputs;
        const std::string prefix = "input_";
        for (int i = 0; i < info.nb_inputs; ++i) {
            auto in_name = prefix + std::to_string(i);
            g->getNodeOrCreatePlaceHolder(in_name);
            inputs.push_back(in_name);
        }
        return inputs;
    }

    std::vector<std::string> CrossAttentionRewriter::GetOutputs(std::shared_ptr<AnchorGraph> in, const CrossAttentionPatternInfo &info) {
        std::vector<std::string> outputs;
        auto out_node = in->getNodeByTensorName(info.output_node_name);
        const std::string prefix = out_node->name() + "__attention__";
        for (int i = 0; i < info.nb_outputs; ++i) {
            auto out_name = prefix + std::to_string(i);
            outputs.push_back(out_name);
        }
        return outputs;
    }

}  // namespace optimizer

}  // namespace TNN_NS
