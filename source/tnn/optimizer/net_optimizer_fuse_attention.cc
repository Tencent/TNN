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

#include "tnn/optimizer/net_optimizer_fuse_attention.h"

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

    NetOptimizerRegister<NetOptimizerFuseAttention> g_net_optimizer_fuse_attention(OptPriority::P1);

    std::string NetOptimizerFuseAttention::Strategy() {
        return kNetOptimizerFuseAttention;
    }

    bool NetOptimizerFuseAttention::IsSupported(const NetworkConfig &net_config) {
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

    struct AttentionPattenInfo {
        std::string graph_str;
        int nb_inputs;
        int nb_outputs;
        std::string shape_node_name;
        std::string div_node_name;
        std::string output_node_name;
        int cast_mask_ = -1;
        bool has_attention_mask = true;
    };

    class AttentionRewriter {
        public:
            AttentionRewriter(std::shared_ptr<Graph> graph, NetStructure *structure, NetResource *resource)
                : graph_(graph), structure_(structure), resource_(resource) {
            }

            Status Rewrite(const AttentionPattenInfo &patten_info);

        private:
            Status GetHeadSize(std::shared_ptr<AnchorGraph> in, const AttentionPattenInfo &info);
            Status GetQScaling(std::shared_ptr<AnchorGraph> in, const AttentionPattenInfo &info);
            MatMulLayerResource *GetWeight(std::shared_ptr<AnchorGraph> in, const std::string &mm_node_name);
            EltwiseLayerResource *GetBias(std::shared_ptr<TNN_NS::AnchorGraph> in, const std::string &add_node_name);

            std::vector<std::string> GetInputs(std::shared_ptr<Graph> g, const AttentionPattenInfo &info);
            std::vector<std::string> GetOutputs(std::shared_ptr<AnchorGraph> in, const AttentionPattenInfo &info);
            Status ModifyIOBinding(std::shared_ptr<Graph> g, const AttentionPattenInfo &info,
                                   std::vector<std::string> &in_names, std::vector<std::string> &out_names);

            std::shared_ptr<Graph> graph_;
            NetStructure *structure_;
            NetResource *resource_;

            int head_num_;
            int size_per_head_;
            int hidden_size_;
            float q_scaling_;
    };

    Status NetOptimizerFuseAttention::Optimize(NetStructure *structure, NetResource *resource) {
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

        AttentionRewriter rewriter(graph, structure, resource);

        for (const auto &patten : GetAttentionPattens()) {
            RETURN_ON_FAIL(rewriter.Rewrite(patten));
        }

        // TNN_NS::Logger::instance().set_verbose_level("W");
        RETURN_ON_FAIL(EliminateRedundantCasts(structure, resource));

        return TNN_OK;
    }

    Status NetOptimizerFuseAttention::EliminateRedundantCasts(NetStructure *structure, NetResource *resource) {
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

    std::vector<AttentionPattenInfo> NetOptimizerFuseAttention::GetAttentionPattens() {
        std::vector<AttentionPattenInfo> pattens;

        // Bert
        {
            AttentionPattenInfo bert_patten;
            bert_patten.graph_str = R"(
                graph(%in, %attn_mask, %num_heads, %per_head_size, %hidden_size):
                    %q_linear_mm          = MatMul(%in)
                    %q_linear_add         = Add(%q_linear_mm)
                    %k_linear_mm          = MatMul(%in)
                    %k_linear_add         = Add(%k_linear_mm)
                    %v_linear_mm          = MatMul(%in)
                    %v_linear_add         = Add(%v_linear_mm)
                    %q_batch_shape        = Shape(%q_linear_add)
                    %q_batch_gather       = Gather(%q_batch_shape)
                    %q_batch              = Unsqueeze(%q_batch_gather)
                    %q_seqlen_shape       = Shape(%q_linear_add)
                    %q_seqlen_gather      = Gather(%q_seqlen_shape)
                    %q_seqlen             = Unsqueeze(%q_seqlen_gather)
                    %q_reshape_shape      = Concat(%q_batch, %q_seqlen, %num_heads, %per_head_size)
                    %q_reshape            = ReshapeTorch(%q_linear_add, %q_reshape_shape)
                    %q_permute            = Permute(%q_reshape)
                    %k_batch_shape        = Shape(%k_linear_add)
                    %k_batch_gather       = Gather(%k_batch_shape)
                    %k_batch              = Unsqueeze(%k_batch_gather)
                    %k_seqlen_shape       = Shape(%k_linear_add)
                    %k_seqlen_gather      = Gather(%k_seqlen_shape)
                    %k_seqlen             = Unsqueeze(%k_seqlen_gather)
                    %k_reshape_shape      = Concat(%k_batch, %k_seqlen, %num_heads, %per_head_size)
                    %k_reshape            = ReshapeTorch(%k_linear_add, %k_reshape_shape)
                    %k_permute            = Permute(%k_reshape)
                    %v_batch_shape        = Shape(%v_linear_add)
                    %v_batch_gather       = Gather(%v_batch_shape)
                    %v_batch              = Unsqueeze(%v_batch_gather)
                    %v_seqlen_shape       = Shape(%v_linear_add)
                    %v_seqlen_gather      = Gather(%v_seqlen_shape)
                    %v_seqlen             = Unsqueeze(%v_seqlen_gather)
                    %v_reshape_shape      = Concat(%v_batch, %v_seqlen, %num_heads, %per_head_size)
                    %v_reshape            = ReshapeTorch(%v_linear_add, %v_reshape_shape)
                    %v_permute            = Permute(%v_reshape)
                    %k_permute_trans      = PermuteV2(%k_permute)
                    %attn_score           = MatMul(%q_permute, %k_permute_trans)
                    %attn_score_div       = Div(%attn_score)
                    %attn_score_mask      = Add(%attn_score_div, %attn_mask)
                    %attn_score_softmax   = SoftmaxCaffe(%attn_score_mask)
                    %attn_context         = MatMul(%attn_score_softmax, %v_permute)
                    %attn_context_permute = Permute(%attn_context)
                    %ac_batch_shape       = Shape(%attn_context_permute)
                    %ac_batch_gather      = Gather(%ac_batch_shape)
                    %ac_batch             = Unsqueeze(%ac_batch_gather)
                    %ac_seqlen_shape      = Shape(%attn_context_permute)
                    %ac_seqlen_gather     = Gather(%ac_seqlen_shape)
                    %ac_seqlen            = Unsqueeze(%ac_seqlen_gather)
                    %ac_reshape_shape     = Concat(%ac_batch, %ac_seqlen, %hidden_size)
                    %attn_context_reshape = ReshapeTorch(%attn_context_permute, %ac_reshape_shape)
                    %o_linear_mm          = MatMul(%attn_context_reshape)
                    return (%o_linear_mm)
            )";
            bert_patten.nb_inputs        = 5;
            bert_patten.nb_outputs       = 1;
            bert_patten.shape_node_name  = "@q_reshape_shape";
            bert_patten.div_node_name    = "@attn_score_div";
            bert_patten.output_node_name = "@o_linear_mm";
            pattens.push_back(bert_patten);
        }

        // DistilBert
        {
            AttentionPattenInfo distil_bert_patten;
            distil_bert_patten.graph_str = R"(
                graph(%in, %attn_mask, %num_heads, %per_head_size, %hidden_size, %minus_one, %one):
                    %batch_shape          = Shape(%in)
                    %batch_gather         = Gather(%batch_shape)
                    %batch                = Unsqueeze(%batch_gather)
                    %seqlen_shape         = Shape(%in)
                    %seqlen_gather        = Gather(%seqlen_shape)
                    %seqlen               = Unsqueeze(%seqlen_gather)
                    %q_linear_mm          = MatMul(%in)
                    %q_linear_add         = Add(%q_linear_mm)
                    %reshape_shape        = Concat(%batch, %minus_one, %num_heads, %per_head_size)
                    %q_reshape            = ReshapeTorch(%q_linear_add, %reshape_shape)
                    %q_permute            = PermuteV2(%q_reshape)
                    %k_linear_mm          = MatMul(%in)
                    %k_linear_add         = Add(%k_linear_mm)
                    %k_reshape            = ReshapeTorch(%k_linear_add, %reshape_shape)
                    %k_permute            = PermuteV2(%k_reshape)
                    %v_linear_mm          = MatMul(%in)
                    %v_linear_add         = Add(%v_linear_mm)
                    %v_reshape            = ReshapeTorch(%v_linear_add, %reshape_shape)
                    %v_permute            = PermuteV2(%v_reshape)
                    %q_permute_div        = Div(%q_permute)
                    %k_permute_trans      = PermuteV2(%k_permute)
                    %attn_score           = MatMul(%q_permute_div, %k_permute_trans)
                    %mask_reshape_shape   = Concat(%batch, %one, %one, %seqlen)
                    %mask_reshape         = ReshapeTorch(%attn_mask, %mask_reshape_shape)
                    %attn_score_shape     = Shape(%attn_score)
                    %mask_expand          = Expand(%mask_reshape, %attn_score_shape)
                    %attn_score_mask      = Where(%attn_score, %mask_expand)
                    %attn_score_softmax   = SoftmaxCaffe(%attn_score_mask)
                    %attn_context         = MatMul(%attn_score_softmax, %v_permute)
                    %attn_context_permute = PermuteV2(%attn_context)
                    %ac_reshape_shape     = Concat(%batch, %minus_one, %hidden_size)
                    %attn_context_reshape = ReshapeTorch(%attn_context_permute, %ac_reshape_shape)
                    %o_linear_mm          = MatMul(%attn_context_reshape)
                    return (%o_linear_mm)
            )";
            distil_bert_patten.nb_inputs        = 7;
            distil_bert_patten.nb_outputs       = 1;
            distil_bert_patten.shape_node_name  = "@reshape_shape";
            distil_bert_patten.div_node_name    = "@q_permute_div";
            distil_bert_patten.output_node_name = "@o_linear_mm";
            distil_bert_patten.cast_mask_       = 1;
            pattens.push_back(distil_bert_patten);
        }

        // Lxmert
        {
            AttentionPattenInfo lxmert_patten;
            lxmert_patten.graph_str = R"(
                graph(%in, %num_heads, %per_head_size, %hidden_size):
                    %q_linear_mm          = MatMul(%in)
                    %q_linear_add         = Add(%q_linear_mm)
                    %k_linear_mm          = MatMul(%in)
                    %k_linear_add         = Add(%k_linear_mm)
                    %v_linear_mm          = MatMul(%in)
                    %v_linear_add         = Add(%v_linear_mm)
                    %q_batch_shape        = Shape(%q_linear_add)
                    %q_batch_gather       = Gather(%q_batch_shape)
                    %q_batch              = Unsqueeze(%q_batch_gather)
                    %q_seqlen_shape       = Shape(%q_linear_add)
                    %q_seqlen_gather      = Gather(%q_seqlen_shape)
                    %q_seqlen             = Unsqueeze(%q_seqlen_gather)
                    %q_reshape_shape      = Concat(%q_batch, %q_seqlen, %num_heads, %per_head_size)
                    %q_reshape            = ReshapeTorch(%q_linear_add, %q_reshape_shape)
                    %q_permute            = Permute(%q_reshape)
                    %k_batch_shape        = Shape(%k_linear_add)
                    %k_batch_gather       = Gather(%k_batch_shape)
                    %k_batch              = Unsqueeze(%k_batch_gather)
                    %k_seqlen_shape       = Shape(%k_linear_add)
                    %k_seqlen_gather      = Gather(%k_seqlen_shape)
                    %k_seqlen             = Unsqueeze(%k_seqlen_gather)
                    %k_reshape_shape      = Concat(%k_batch, %k_seqlen, %num_heads, %per_head_size)
                    %k_reshape            = ReshapeTorch(%k_linear_add, %k_reshape_shape)
                    %k_permute            = Permute(%k_reshape)
                    %v_batch_shape        = Shape(%v_linear_add)
                    %v_batch_gather       = Gather(%v_batch_shape)
                    %v_batch              = Unsqueeze(%v_batch_gather)
                    %v_seqlen_shape       = Shape(%v_linear_add)
                    %v_seqlen_gather      = Gather(%v_seqlen_shape)
                    %v_seqlen             = Unsqueeze(%v_seqlen_gather)
                    %v_reshape_shape      = Concat(%v_batch, %v_seqlen, %num_heads, %per_head_size)
                    %v_reshape            = ReshapeTorch(%v_linear_add, %v_reshape_shape)
                    %v_permute            = Permute(%v_reshape)
                    %k_permute_trans      = PermuteV2(%k_permute)
                    %attn_score           = MatMul(%q_permute, %k_permute_trans)
                    %attn_score_div       = Div(%attn_score)
                    %attn_score_softmax   = SoftmaxCaffe(%attn_score_div)
                    %attn_context         = MatMul(%attn_score_softmax, %v_permute)
                    %attn_context_permute = Permute(%attn_context)
                    %ac_batch_shape       = Shape(%attn_context_permute)
                    %ac_batch_gather      = Gather(%ac_batch_shape)
                    %ac_batch             = Unsqueeze(%ac_batch_gather)
                    %ac_seqlen_shape      = Shape(%attn_context_permute)
                    %ac_seqlen_gather     = Gather(%ac_seqlen_shape)
                    %ac_seqlen            = Unsqueeze(%ac_seqlen_gather)
                    %ac_reshape_shape     = Concat(%ac_batch, %ac_seqlen, %hidden_size)
                    %attn_context_reshape = ReshapeTorch(%attn_context_permute, %ac_reshape_shape)
                    %o_linear_mm          = MatMul(%attn_context_reshape)
                    return (%o_linear_mm)
            )";
            lxmert_patten.nb_inputs          = 4;
            lxmert_patten.nb_outputs         = 1;
            lxmert_patten.has_attention_mask = false;
            lxmert_patten.shape_node_name    = "@q_reshape_shape";
            lxmert_patten.div_node_name      = "@attn_score_div";
            lxmert_patten.output_node_name   = "@o_linear_mm";
            pattens.push_back(lxmert_patten);
        }

        // Albert
        {
            AttentionPattenInfo albert_patten;
            albert_patten.graph_str = R"(
                graph(%in, %attn_mask, %num_heads, %per_head_size):
                    %q_linear_mm          = MatMul(%in)
                    %q_linear_add         = Add(%q_linear_mm)
                    %k_linear_mm          = MatMul(%in)
                    %k_linear_add         = Add(%k_linear_mm)
                    %v_linear_mm          = MatMul(%in)
                    %v_linear_add         = Add(%v_linear_mm)
                    %q_batch_shape        = Shape(%q_linear_add)
                    %q_batch_gather       = Gather(%q_batch_shape)
                    %q_batch              = Unsqueeze(%q_batch_gather)
                    %q_seqlen_shape       = Shape(%q_linear_add)
                    %q_seqlen_gather      = Gather(%q_seqlen_shape)
                    %q_seqlen             = Unsqueeze(%q_seqlen_gather)
                    %q_reshape_shape      = Concat(%q_batch, %q_seqlen, %num_heads, %per_head_size)
                    %q_reshape            = ReshapeTorch(%q_linear_add, %q_reshape_shape)
                    %q_permute            = Permute(%q_reshape)
                    %k_batch_shape        = Shape(%k_linear_add)
                    %k_batch_gather       = Gather(%k_batch_shape)
                    %k_batch              = Unsqueeze(%k_batch_gather)
                    %k_seqlen_shape       = Shape(%k_linear_add)
                    %k_seqlen_gather      = Gather(%k_seqlen_shape)
                    %k_seqlen             = Unsqueeze(%k_seqlen_gather)
                    %k_reshape_shape      = Concat(%k_batch, %k_seqlen, %num_heads, %per_head_size)
                    %k_reshape            = ReshapeTorch(%k_linear_add, %k_reshape_shape)
                    %k_permute            = Permute(%k_reshape)
                    %v_batch_shape        = Shape(%v_linear_add)
                    %v_batch_gather       = Gather(%v_batch_shape)
                    %v_batch              = Unsqueeze(%v_batch_gather)
                    %v_seqlen_shape       = Shape(%v_linear_add)
                    %v_seqlen_gather      = Gather(%v_seqlen_shape)
                    %v_seqlen             = Unsqueeze(%v_seqlen_gather)
                    %v_reshape_shape      = Concat(%v_batch, %v_seqlen, %num_heads, %per_head_size)
                    %v_reshape            = ReshapeTorch(%v_linear_add, %v_reshape_shape)
                    %v_permute            = Permute(%v_reshape)
                    %k_permute_trans      = PermuteV2(%k_permute)
                    %attn_score           = MatMul(%q_permute, %k_permute_trans)
                    %attn_score_div       = Div(%attn_score)
                    %attn_score_mask      = Add(%attn_score_div, %attn_mask)
                    %attn_score_softmax   = SoftmaxCaffe(%attn_score_mask)
                    %attn_context         = MatMul(%attn_score_softmax, %v_permute)
                    %attn_context_permute = PermuteV2(%attn_context)
                    %attn_context_reshape = FlattenTorch(%attn_context_permute)
                    %o_linear_mm          = MatMul(%attn_context_reshape)
                    return (%o_linear_mm)
            )";
            albert_patten.nb_inputs        = 4;
            albert_patten.nb_outputs       = 1;
            albert_patten.shape_node_name  = "@q_reshape_shape";
            albert_patten.div_node_name    = "@attn_score_div";
            albert_patten.output_node_name = "@o_linear_mm";
            pattens.push_back(albert_patten);
        }

        return pattens;
    }

    Status AttentionRewriter::Rewrite(const AttentionPattenInfo &patten_info) {
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

            if (GetHeadSize(in, patten_info) != TNN_OK) {
                return nullptr;
            }

            if (GetQScaling(in, patten_info) != TNN_OK) {
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

            auto add_q = GetBias(in, "@q_linear_add");
            auto add_k = GetBias(in, "@k_linear_add");
            auto add_v = GetBias(in, "@v_linear_add");
            if (!add_q || !add_k || !add_v) {
                WARN("bias resource is nil");
                return nullptr;
            }

            auto g         = std::make_shared<Graph>();
            auto in_names  = GetInputs(g, patten_info);
            auto out_names = GetOutputs(in, patten_info);
            if (ModifyIOBinding(g, patten_info, in_names, out_names) != TNN_OK) {
                return nullptr;
            }
            auto status = g->createNode(LAYER_FUSED, in_names, out_names);
            if (status != TNN_OK) {
                return nullptr;
            }

            auto fused_node = g->getNodeByTensorName(out_names[0]);

            auto fused_param                     = std::make_shared<FusedLayerParam>();
            fused_param->type                    = FusionType_Attention;
            fused_param->attention_head_num      = head_num_;
            fused_param->attention_size_per_head = size_per_head_;
            fused_param->attention_q_scaling     = q_scaling_;
            fused_param->has_attention_mask      = patten_info.has_attention_mask;
            fused_node->info->param              = fused_param;

            auto fused_resource = std::make_shared<FusedLayerResource>();
            if (resource_->resource_map.find(out_names[0]) != resource_->resource_map.end()) {
                WARN("fused_resource name conflict");
                return nullptr;
            }
            fused_resource->attention_q_mm        = *matmul_q;
            fused_resource->attention_k_mm        = *matmul_k;
            fused_resource->attention_v_mm        = *matmul_v;
            fused_resource->attention_o_mm        = *matmul_o;
            fused_resource->attention_q_bias      = *add_q;
            fused_resource->attention_k_bias      = *add_k;
            fused_resource->attention_v_bias      = *add_v;
            resource_->resource_map[out_names[0]] = fused_resource;

            return g;
        };

        RETURN_ON_FAIL(graph_->rewrite(pattern, gen));

        return TNN_OK;
    }

    Status AttentionRewriter::GetHeadSize(std::shared_ptr<AnchorGraph> in, const AttentionPattenInfo &info) {
        auto reshape_shape_node = in->getNodeByTensorName(info.shape_node_name);
        if (!reshape_shape_node) {
            WARN("reshape node not found");
            return Status(TNNERR_NET_ERR, "reshape node not found");
        }
        if (reshape_shape_node->info->inputs.size() != 4) {
            WARN("reshape node inputs size error");
            return Status(TNNERR_NET_ERR, "reshape node inputs size error");
        }
        if (resource_->constant_map.find(reshape_shape_node->info->inputs[2]) == resource_->constant_map.end() ||
            resource_->constant_map.find(reshape_shape_node->info->inputs[3]) == resource_->constant_map.end()) {
            WARN("reshape node input not found in constant_map");
            return Status(TNNERR_NET_ERR, "reshape node input not found in constant_map");
        }
        head_num_      = resource_->constant_map[reshape_shape_node->info->inputs[2]]->force_to<int*>()[0];
        size_per_head_ = resource_->constant_map[reshape_shape_node->info->inputs[3]]->force_to<int*>()[0];

        hidden_size_ = head_num_ * size_per_head_;
        return TNN_OK;
    }

    Status AttentionRewriter::GetQScaling(std::shared_ptr<AnchorGraph> in, const AttentionPattenInfo &info) {
        auto attn_score_div_node = in->getNodeByTensorName(info.div_node_name);
        if (!attn_score_div_node) {
            WARN("div node not found");
            return Status(TNNERR_NET_ERR, "div node not found");
        }
        if (attn_score_div_node->info->inputs.size() != 1) {
            WARN("div node inputs size error");
            return Status(TNNERR_NET_ERR, "div node inputs size error");
        }
        if (resource_->resource_map.find(attn_score_div_node->info->name) == resource_->resource_map.end()) {
            WARN("div node resource not found in resource_map");
            return Status(TNNERR_NET_ERR, "div node resource not found in resource_map");
        }
        auto div_layer_res = dynamic_cast<EltwiseLayerResource *>(resource_->resource_map[attn_score_div_node->info->name].get());
        if (!div_layer_res) {
            WARN("div node resource is nil");
            return Status(TNNERR_NET_ERR, "div node resource is nil");
        }
        const float denom = div_layer_res->element_handle.force_to<float*>()[0];
        q_scaling_ = denom / sqrt(size_per_head_); // denom = sqrt(size_per_head) * q_scaling
        return TNN_OK;
    }

    MatMulLayerResource *AttentionRewriter::GetWeight(std::shared_ptr<AnchorGraph> in, const std::string &mm_node_name) {
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
        if (matmul_weight_dims[0] != hidden_size_ || matmul_weight_dims[1] != hidden_size_) {
            WARN("matmul_resource shape not supported");
            return nullptr;
        }
        return matmul_res;
    }

    EltwiseLayerResource *AttentionRewriter::GetBias(std::shared_ptr<TNN_NS::AnchorGraph> in, const std::string &add_node_name) {
        auto add_node = in->getNodeByTensorName(add_node_name);
        if (!add_node) {
            WARN("node of interest not found");
            return nullptr;
        }
        auto node_info_name = add_node->info->name;
        if (resource_->resource_map.find(node_info_name) == resource_->resource_map.end()) {
            WARN("add_resource is not found");
            return nullptr;
        }
        auto add_res = dynamic_cast<EltwiseLayerResource *>(resource_->resource_map[node_info_name].get());
        if (!add_res) {
            WARN("add_resource is nil");
            return nullptr;
        }
        auto add_bias_dims = add_res->element_handle.GetBufferDims();
        if (add_bias_dims.size() != 1) {
            WARN("add_resource dims not support");
            return nullptr;
        }
        if (add_bias_dims[0] != hidden_size_) {
            WARN("add_resource shape not supported");
            return nullptr;
        }
        return add_res;
    }

    std::vector<std::string> AttentionRewriter::GetInputs(std::shared_ptr<Graph> g, const AttentionPattenInfo &info) {
        std::vector<std::string> inputs;
        const std::string prefix = "input_";
        for (int i = 0; i < info.nb_inputs; ++i) {
            auto in_name = prefix + std::to_string(i);
            g->getNodeOrCreatePlaceHolder(in_name);
            inputs.push_back(in_name);
        }
        return inputs;
    }

    std::vector<std::string> AttentionRewriter::GetOutputs(std::shared_ptr<AnchorGraph> in, const AttentionPattenInfo &info) {
        std::vector<std::string> outputs;
        auto out_node = in->getNodeByTensorName(info.output_node_name);
        const std::string prefix = out_node->name() + "__attention__";
        for (int i = 0; i < info.nb_outputs; ++i) {
            auto out_name = prefix + std::to_string(i);
            outputs.push_back(out_name);
        }
        return outputs;
    }

    Status AttentionRewriter::ModifyIOBinding(std::shared_ptr<Graph> g, const AttentionPattenInfo &info,
                                              std::vector<std::string> &in_names, std::vector<std::string> &out_names) {
        if (info.cast_mask_ >= 0 && info.cast_mask_ < in_names.size()) {
            auto cast_name = out_names[0] + "_cast";
            g->createNode(LAYER_CAST, {in_names[info.cast_mask_]}, {cast_name});
            auto cast_node         = g->getNodeByTensorName(cast_name);
            auto cast_param        = std::make_shared<CastLayerParam>();
            cast_param->to         = 1;
            cast_node->info->param = cast_param;
            in_names[info.cast_mask_] = cast_name;
        }
        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
