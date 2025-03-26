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

#include "tnn/optimizer/net_optimizer_effective_transformer.h"

#include <map>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>

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

    NetOptimizerRegister<NetOptimizerEffectiveTransformer> g_net_optimizer_effective_transformer(OptPriority::P2);

    std::string NetOptimizerEffectiveTransformer::Strategy() {
        return kNetOptimizerEffectiveTransformer;
    }

    bool NetOptimizerEffectiveTransformer::IsSupported(const NetworkConfig &net_config) {
        auto device = net_config.device_type;
        if (device == DEVICE_CUDA) {
            return true;
        }
        return false;
    }

    Status NetOptimizerEffectiveTransformer::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }
        // TNN_NS::Logger::instance().set_verbose_level("D");

        std::shared_ptr<Graph> graph = std::make_shared<Graph>();
        RETURN_ON_FAIL(graph->fromInterpreted(structure, resource));

        RETURN_ON_FAIL(OptimizeForAttention(graph));
        RETURN_ON_FAIL(OptimizeForFFN(graph));
        RETURN_ON_FAIL(EliminateRedundantReformats(structure, resource));
        RETURN_ON_FAIL(ReorderDenseOps(structure, resource));

        // ModelPacker packer(structure, resource);
        // packer.Pack("pack.tnnproto", "pack.tnnmodel");

        return TNN_OK;
    }

#define TNN_GRAPH_PREPARE_NODE(name)                                    \
    auto name## _name = #name;                                          \
    auto name## _node = g->getNodeOrCreatePlaceHolder(name## _name);

    static std::vector<std::string> RemovePadding(std::shared_ptr<Graph> g, const std::vector<std::string> &in_names, const std::string &layer_name) {
        std::vector<std::string> out_names = {layer_name + std::string("__eff_to_dense__"), "pad_offset", "trt_offset"};
        auto status = g->createNode(LAYER_EFFECTIVE_TRANSFORMER, in_names, out_names);
        if (status != TNN_OK) {
            return {};
        }
        auto remove_pad_node = g->getNodeByTensorName(out_names[0]);
        auto remove_pad_param = std::make_shared<EffectiveTransformerLayerParam>();
        remove_pad_param->is_remove_padding = true;
        remove_pad_node->info->param = remove_pad_param;
        return out_names;
    }

    static std::vector<std::string> RebuildPadding(std::shared_ptr<Graph> g, const std::vector<std::string> &in_names, const std::string &layer_name) {
        std::vector<std::string> out_names = {layer_name + std::string("__eff_to_sparse__")};
        auto status = g->createNode(LAYER_EFFECTIVE_TRANSFORMER, in_names, out_names);
        if (status != TNN_OK) {
            return {};
        }
        auto rebuild_pad_node = g->getNodeByTensorName(out_names[0]);
        auto rebuild_pad_param = std::make_shared<EffectiveTransformerLayerParam>();
        rebuild_pad_param->is_remove_padding = false;
        rebuild_pad_node->info->param = rebuild_pad_param;
        return out_names;
    }

    struct FusedAttentionPattenInfo {
        std::string graph_str;
        std::string graph_str_with_layernorm;
        int nb_inputs;
        int nb_outputs;
        std::string att_node_name;
        std::string ln_node_name;
    };

    class FusedAttentionRewriter {
        public:
            FusedAttentionRewriter(std::shared_ptr<Graph> graph)
                : graph_(graph) {
            }
            Status Rewrite(const FusedAttentionPattenInfo &patten_info);
            Status RewriteWithLayerNorm(const FusedAttentionPattenInfo &patten_info);
        private:
            std::vector<std::string> GetInputs(std::shared_ptr<Graph> g, const FusedAttentionPattenInfo &info);
            std::shared_ptr<Graph> graph_;
    };

    std::vector<std::string> FusedAttentionRewriter::GetInputs(std::shared_ptr<Graph> g, const FusedAttentionPattenInfo &info) {
        std::vector<std::string> inputs;
        const std::string prefix = "input_";
        for (int i = 0; i < info.nb_inputs; ++i) {
            auto in_name = prefix + std::to_string(i);
            g->getNodeOrCreatePlaceHolder(in_name);
            inputs.push_back(in_name);
        }
        return inputs;
    }

    Status FusedAttentionRewriter::Rewrite(const FusedAttentionPattenInfo &patten_info) {
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

            auto attention_node = in->getNodeByTensorName(patten_info.att_node_name);
            if (!attention_node) {
                WARN("node of interest not found");
                return nullptr;
            }
            auto att_name = attention_node->name();

            auto g = std::make_shared<Graph>();
            auto in_names  = GetInputs(g, patten_info);

            std::vector<std::string> dense_outs = RemovePadding(g, in_names, att_name);
            if (dense_outs.size() != 3) {
                WARN("create remove padding node failed");
                return nullptr;
            }

            // in_names[1]: attention mask
            auto status = g->createNode(LAYER_FUSED, {dense_outs[0], in_names[1], dense_outs[2], dense_outs[1]}, {att_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_attention_node = g->getNodeByTensorName(att_name);
            new_attention_node->info->param = attention_node->info->param->Copy();
            auto attention_param = dynamic_cast<FusedLayerParam *>(new_attention_node->info->param.get());
            if (!attention_param) {
                WARN("attention_param is nil");
                return nullptr;
            }
            if (attention_param->type != FusionType_Attention) {
                WARN("type is not attention layer");
                return nullptr;
            }
            attention_param->dense_mode = true;

            std::vector<std::string> sparse_outs = RebuildPadding(g, {att_name, dense_outs[1]}, att_name);
            if (sparse_outs.size() != 1) {
                WARN("create rebuild padding node failed");
                return nullptr;
            }

            return g;
        };

        RETURN_ON_FAIL(graph_->rewrite(pattern, gen));

        return TNN_OK;
    }

    Status FusedAttentionRewriter::RewriteWithLayerNorm(const FusedAttentionPattenInfo &patten_info) {
        GraphParser parser;
        std::shared_ptr<Graph> pattern = nullptr;
        if (parser.parseFromString(patten_info.graph_str_with_layernorm)) {
            pattern = parser.getGraph();
        } else {
            return Status(TNNERR_PARAM_ERR, "invalid pattern syntax.");
        }

        auto gen = [&](std::shared_ptr<AnchorGraph> in) -> std::shared_ptr<Graph> {
            if (in->inputs().size() != patten_info.nb_inputs + 2 || in->outputs().size() != patten_info.nb_outputs) {
                return nullptr;
            }

            auto attention_node = in->getNodeByTensorName(patten_info.att_node_name);
            if (!attention_node) {
                WARN("node of interest not found");
                return nullptr;
            }
            auto att_name = attention_node->name();

            auto ln_node = in->getNodeByTensorName(patten_info.ln_node_name);
            if (!ln_node) {
                WARN("node of interest not found");
                return nullptr;
            }
            auto ln_name = ln_node->name();

            auto g        = std::make_shared<Graph>();
            auto in_names = GetInputs(g, patten_info);
            TNN_GRAPH_PREPARE_NODE(scale);
            TNN_GRAPH_PREPARE_NODE(bias);

            std::vector<std::string> dense_outs = RemovePadding(g, in_names, att_name);
            if (dense_outs.size() != 3) {
                WARN("create remove padding node failed");
                return nullptr;
            }

            // in_names[1]: attention mask
            auto status = g->createNode(LAYER_FUSED, {dense_outs[0], in_names[1], dense_outs[2], dense_outs[1]}, {att_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_attention_node = g->getNodeByTensorName(att_name);
            new_attention_node->info->param = attention_node->info->param->Copy();
            auto attention_param = dynamic_cast<FusedLayerParam *>(new_attention_node->info->param.get());
            if (!attention_param) {
                WARN("attention_param is nil");
                return nullptr;
            }
            if (attention_param->type != FusionType_Attention) {
                WARN("type is not attention layer");
                return nullptr;
            }

            status = g->createNode(LAYER_FUSED, {att_name, dense_outs[0], scale_name, bias_name}, {ln_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_ln_node = g->getNodeByTensorName(ln_name);
            new_ln_node->info->param = ln_node->info->param->Copy();
            auto ln_param = dynamic_cast<FusedLayerParam *>(new_ln_node->info->param.get());
            if (!ln_param) {
                WARN("ln_param is nil");
                return nullptr;
            }
            if (ln_param->type != FusionType_AddBiasResidualLayerNorm) {
                WARN("type is not layernorm layer");
                return nullptr;
            }

            std::vector<std::string> sparse_outs = RebuildPadding(g, {ln_name, dense_outs[1]}, ln_name);
            if (sparse_outs.size() != 1) {
                WARN("create rebuild padding node failed");
                return nullptr;
            }

            return g;
        };

        RETURN_ON_FAIL(graph_->rewrite(pattern, gen));

        return TNN_OK;
    }

    std::vector<FusedAttentionPattenInfo> GetFusedAttentionPattens() {
        std::vector<FusedAttentionPattenInfo> pattens;

        // Bert
        {
            FusedAttentionPattenInfo bert_patten;
            bert_patten.graph_str = R"(
                graph(%in, %attn_mask, %num_heads, %per_head_size, %hidden_size):
                    %att_out = Fused(%in, %attn_mask, %num_heads, %per_head_size, %hidden_size)
                    return (%att_out)
            )";
            bert_patten.graph_str_with_layernorm = R"(
                graph(%in, %attn_mask, %num_heads, %per_head_size, %hidden_size, %scale, %bias):
                    %dense, %pad_offset, %trt_offset = EffectiveTransformer(%in, %attn_mask, %num_heads, %per_head_size, %hidden_size)
                    %att_out                         = Fused(%dense, %attn_mask, %trt_offset, %pad_offset)
                    %sparse                          = EffectiveTransformer(%att_out, %pad_offset)
                    %ln_out                          = Fused(%sparse, %in, %scale, %bias)
                    return (%ln_out)
            )";
            bert_patten.nb_inputs     = 5;
            bert_patten.nb_outputs    = 1;
            bert_patten.att_node_name = "@att_out";
            bert_patten.ln_node_name  = "@ln_out";
            pattens.push_back(bert_patten);
        }

        // DistilBert
        {
            FusedAttentionPattenInfo distil_bert_patten;
            distil_bert_patten.graph_str = R"(
                graph(%in, %attn_mask, %num_heads, %per_head_size, %hidden_size, %minus_one, %one):
                    %att_out = Fused(%in, %attn_mask, %num_heads, %per_head_size, %hidden_size, %minus_one, %one)
                    return (%att_out)
            )";
            distil_bert_patten.graph_str_with_layernorm = R"(
                graph(%in, %attn_mask, %num_heads, %per_head_size, %hidden_size, %minus_one, %one, %scale, %bias):
                    %dense, %pad_offset, %trt_offset = EffectiveTransformer(%in, %attn_mask, %num_heads, %per_head_size, %hidden_size, %minus_one, %one)
                    %att_out                         = Fused(%dense, %attn_mask, %trt_offset, %pad_offset)
                    %sparse                          = EffectiveTransformer(%att_out, %pad_offset)
                    %ln_out                          = Fused(%sparse, %in, %scale, %bias)
                    return (%ln_out)
            )";
            distil_bert_patten.nb_inputs     = 7;
            distil_bert_patten.nb_outputs    = 1;
            distil_bert_patten.att_node_name = "@att_out";
            distil_bert_patten.ln_node_name  = "@ln_out";
            pattens.push_back(distil_bert_patten);
        }

        // Albert
        {
            FusedAttentionPattenInfo albert_patten;
            albert_patten.graph_str = R"(
                graph(%in, %attn_mask, %num_heads, %per_head_size):
                    %att_out = Fused(%in, %attn_mask, %num_heads, %per_head_size)
                    return (%att_out)
            )";
            albert_patten.graph_str_with_layernorm = R"(
                graph(%in, %attn_mask, %num_heads, %per_head_size, %scale, %bias):
                    %dense, %pad_offset, %trt_offset = EffectiveTransformer(%in, %attn_mask, %num_heads, %per_head_size)
                    %att_out                         = Fused(%dense, %attn_mask, %trt_offset, %pad_offset)
                    %sparse                          = EffectiveTransformer(%att_out, %pad_offset)
                    %ln_out                          = Fused(%sparse, %in, %scale, %bias)
                    return (%ln_out)
            )";
            albert_patten.nb_inputs     = 4;
            albert_patten.nb_outputs    = 1;
            albert_patten.att_node_name = "@att_out";
            albert_patten.ln_node_name  = "@ln_out";
            pattens.push_back(albert_patten);
        }

        return pattens;
    }

    Status NetOptimizerEffectiveTransformer::OptimizeForAttention(std::shared_ptr<Graph> graph) {
        FusedAttentionRewriter rewriter(graph);

        for (const auto &patten : GetFusedAttentionPattens()) {
            RETURN_ON_FAIL(rewriter.Rewrite(patten));
            RETURN_ON_FAIL(rewriter.RewriteWithLayerNorm(patten));
        }

        return TNN_OK;
    }

    Status NetOptimizerEffectiveTransformer::OptimizeForFFN(std::shared_ptr<Graph> graph) {
        std::string graph_str = R"(
            graph(%dense, %pad_offset, %scale, %bias):
                %sparse  = EffectiveTransformer(%dense, %pad_offset)
                %ffn_out = Fused(%sparse)
                %out     = Fused(%ffn_out, %sparse, %scale, %bias)
                return (%out)
        )";

        GraphParser parser;
        std::shared_ptr<Graph> pattern = nullptr;
        if (parser.parseFromString(graph_str)) {
            pattern = parser.getGraph();
        } else {
            return Status(TNNERR_PARAM_ERR, "invalid pattern syntax.");
        }

        auto gen = [&](std::shared_ptr<AnchorGraph> in) -> std::shared_ptr<Graph> {
            if (in->inputs().size() != 4 || in->outputs().size() != 1 ) {
                return nullptr;
            }

            auto ffn_node = in->getNodeByTensorName(std::string("@ffn_out"));
            if (!ffn_node) {
                WARN("node of interest not found");
                return nullptr;
            }
            auto ffn_name = ffn_node->name();

            auto ln_node = in->getNodeByTensorName(std::string("@out"));
            if (!ln_node) {
                WARN("node of interest not found");
                return nullptr;
            }
            auto ln_name = ln_node->name();

            auto g = std::make_shared<Graph>();

            TNN_GRAPH_PREPARE_NODE(dense);
            TNN_GRAPH_PREPARE_NODE(pad_offset);
            TNN_GRAPH_PREPARE_NODE(scale);
            TNN_GRAPH_PREPARE_NODE(bias);

            auto status = g->createNode(LAYER_FUSED, {dense_name}, {ffn_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_ffn_node = g->getNodeByTensorName(ffn_name);
            new_ffn_node->info->param = ffn_node->info->param->Copy();
            auto ffn_param = dynamic_cast<FusedLayerParam *>(new_ffn_node->info->param.get());
            if (!ffn_param) {
                WARN("ffn_param is nil");
                return nullptr;
            }
            if (ffn_param->type != FusionType_FFN) {
                WARN("type is not ffn layer");
                return nullptr;
            }

            status = g->createNode(LAYER_FUSED, {ffn_name, dense_name, scale_name, bias_name}, {ln_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_ln_node = g->getNodeByTensorName(ln_name);
            new_ln_node->info->param = ln_node->info->param->Copy();
            auto ln_param = dynamic_cast<FusedLayerParam *>(new_ln_node->info->param.get());
            if (!ln_param) {
                WARN("ln_param is nil");
                return nullptr;
            }
            if (ln_param->type != FusionType_AddBiasResidualLayerNorm) {
                WARN("type is not layernorm layer");
                return nullptr;
            }

            std::vector<std::string> sparse_outs = RebuildPadding(g, {ln_name, pad_offset_name}, ln_name);
            if (sparse_outs.size() != 1) {
                WARN("create rebuild padding node failed");
                return nullptr;
            }

            return g;
        };

        RETURN_ON_FAIL(graph->rewrite(pattern, gen));

        return TNN_OK;
    }

    Status NetOptimizerEffectiveTransformer::EliminateRedundantReformats(NetStructure *structure, NetResource *resource) {
        auto ret = Status(TNN_OK);
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        const int count                                     = (const int)layers_orig.size();
        if (count <= 1) {
            return TNN_OK;
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_optimized;
        layers_optimized.push_back(layers_orig[0]);
        for (int index = 1; index < count; index++) {
            layers_optimized.push_back(layers_orig[index]);

            auto layer_info_curr = layers_orig[index];
            auto layer_info_prev = layers_orig[index - 1];

            if (layer_info_curr->type != LAYER_EFFECTIVE_TRANSFORMER ||
                layer_info_prev->type != LAYER_EFFECTIVE_TRANSFORMER) {
                continue;
            }

            auto curr_param = dynamic_cast<EffectiveTransformerLayerParam *>(layer_info_curr->param.get());
            auto prev_param = dynamic_cast<EffectiveTransformerLayerParam *>(layer_info_prev->param.get());
            if (!prev_param || prev_param->is_remove_padding ||
                !curr_param || !curr_param->is_remove_padding) {
                continue;
            }

            if (layer_info_prev->inputs.size() != 2 || layer_info_prev->outputs.size() != 1 ||
                layer_info_curr->inputs.size() <= 2 || layer_info_curr->outputs.size() != 3) {
                LOGE("Error: effective transformer io size error\n");
                return Status(TNNERR_NET_ERR, "Error: effective transformer io size error");
            }
            if (layer_info_prev->outputs[0] != layer_info_curr->inputs[0]) {
                continue;
            }

            auto dense_in   = layer_info_prev->inputs[0];
            auto pad_offset = layer_info_prev->inputs[1];
            std::shared_ptr<LayerInfo> prev_eff_node = nullptr;
            for (const auto & info : layers_optimized) {
                for (const auto & out : info->outputs) {
                    if (out == pad_offset) {
                        prev_eff_node = info;
                        break;
                    }
                }
            }
            if (!prev_eff_node || prev_eff_node->outputs.size() != 3 || prev_eff_node->outputs[1] != pad_offset) {
                LOGE("Error: find prev_eff_node error\n");
                return Status(TNNERR_NET_ERR, "Error: find prev_eff_node error");
            }
            auto trt_offset = prev_eff_node->outputs[2];

            if (layer_info_curr->inputs.size() != prev_eff_node->inputs.size()) {
                continue;
            }
            for (int j = 1; j < layer_info_curr->inputs.size(); ++j) {
                if (layer_info_curr->inputs[j] != prev_eff_node->inputs[j]) {
                    continue;
                }
            }

            std::unordered_map<std::string, std::string> replace_inputs;
            replace_inputs[layer_info_curr->outputs[0]] = dense_in;
            replace_inputs[layer_info_curr->outputs[1]] = pad_offset;
            replace_inputs[layer_info_curr->outputs[2]] = trt_offset;
            for (int j = index; j < count; ++j) {
                auto layer = layers_orig[j];
                for (int k = 0; k < layer->inputs.size(); ++k) {
                    auto input = layer->inputs[k];
                    if (replace_inputs.find(input) != replace_inputs.end()) {
                        layer->inputs[k] = replace_inputs[input];
                    }
                }
            }

            layers_optimized.pop_back();
            layers_optimized.pop_back();
        }
        structure->layers = layers_optimized;

        return ret;
    }

    class NetOptimizerEffectiveTransformer::LayerReorder {
    public:
        LayerReorder(NetStructure *structure);
        void Run();

    private:
        void Union(int x, int y);
        int Find(int x);
        bool IsOrdered(int x, int y);

        NetStructure *structure_;
        int layer_count_;
        std::vector<int> p_;
        std::unordered_map<std::string, int> blob_to_layerid_;
        std::unordered_set<int> dense_to_sparse_idx_;
        std::vector<std::pair<int, int>> dense_ranges_;
    };

    Status NetOptimizerEffectiveTransformer::ReorderDenseOps(NetStructure *structure, NetResource *resource) {
        auto ret = Status(TNN_OK);
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        LayerReorder reorder(structure);
        reorder.Run();

        return ret;
    }

    NetOptimizerEffectiveTransformer::LayerReorder::LayerReorder(NetStructure *structure)
        : structure_(structure) {
        layer_count_ = structure->layers.size();
        p_.resize(layer_count_, -1);
        for (int i = 0; i < layer_count_; ++i) {
            auto layer_info = structure->layers[i];
            if (layer_info->type == LAYER_EFFECTIVE_TRANSFORMER) {
                auto layer_param = dynamic_cast<EffectiveTransformerLayerParam *>(layer_info->param.get());
                if (layer_param && layer_param->is_remove_padding) {
                    p_[i] = i;
                }
                if (layer_param && !layer_param->is_remove_padding) {
                    dense_to_sparse_idx_.insert(i);
                }
            }
            for (const auto &out : layer_info->outputs) {
                blob_to_layerid_[out] = i;
            }
        }
    }

    void NetOptimizerEffectiveTransformer::LayerReorder::Run() {
        for (int i = 0; i < layer_count_; ++i) {
            auto layer_info = structure_->layers[i];
            for (const auto &in : layer_info->inputs) {
                if (blob_to_layerid_.find(in) != blob_to_layerid_.end()) {
                    Union(blob_to_layerid_[in], i);
                }
            }
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_reordered;
        std::unordered_set<int> visited_layers;
        int prev_dense_to_sparse = -1;
        for (int i = 0; i < layer_count_; ++i) {
            if (visited_layers.find(i) != visited_layers.end()) {
                continue;
            }
            int root = p_[i];
            if (root < 0) {
                layers_reordered.push_back(structure_->layers[i]);
            } else {
                int sparse_to_dense = i;
                if (prev_dense_to_sparse > 0 && !IsOrdered(prev_dense_to_sparse, sparse_to_dense)) {
                    auto dense_to_sparse_layer = structure_->layers[prev_dense_to_sparse];
                    auto sparse_to_dense_layer = structure_->layers[sparse_to_dense];
                    auto control_edge = dense_to_sparse_layer->name + "control__";
                    dense_to_sparse_layer->outputs.push_back(control_edge);
                    sparse_to_dense_layer->inputs.push_back(control_edge);
                    structure_->blobs.insert(control_edge);
                }
                int paired_dense_to_sparse = -1;
                for (int j = i + 1; j < layer_count_; ++j) {
                    if (Find(j) == root) {
                        paired_dense_to_sparse = j;
                    }
                }
                prev_dense_to_sparse = paired_dense_to_sparse;
                if (paired_dense_to_sparse < 0) {
                    LOGE("Error: effective transformer operation is not paired\n");
                    return;
                }
                for (int j = sparse_to_dense; j <= paired_dense_to_sparse; ++j) {
                    if (Find(j) < 0) {
                        layers_reordered.push_back(structure_->layers[j]);
                        visited_layers.insert(j);
                    }
                }
                for (int j = sparse_to_dense; j <= paired_dense_to_sparse; ++j) {
                    if (Find(j) == root) {
                        layers_reordered.push_back(structure_->layers[j]);
                        visited_layers.insert(j);
                    }
                }
            }
        }
        structure_->layers = layers_reordered;
    }

    void NetOptimizerEffectiveTransformer::LayerReorder::Union(int x, int y) {
        if (x < 0 || x >= layer_count_ || y <= x || y >= layer_count_) return;
        if (dense_to_sparse_idx_.find(x) != dense_to_sparse_idx_.end()) return;
        if (Find(x) < 0 || Find(y) >= 0) return;
        p_[y] = Find(x);
    }

    int NetOptimizerEffectiveTransformer::LayerReorder::Find(int x) {
        if (p_[x] < 0) return -1;
        if (p_[x] != x) p_[x] = Find(p_[x]);
        return p_[x];
    }

    bool NetOptimizerEffectiveTransformer::LayerReorder::IsOrdered(int x, int y) {
        auto layer_info = structure_->layers[y];
        for (const auto& in : layer_info->inputs) {
            if (blob_to_layerid_.find(in) != blob_to_layerid_.end()) {
                if (blob_to_layerid_[in] == x || IsOrdered(x, blob_to_layerid_[in])) {
                    return true;
                }
            }
        }
        return false;
    }

#undef TNN_GRAPH_PREPARE_NODE

}  // namespace optimizer

}  // namespace TNN_NS
