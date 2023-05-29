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

#include "tnn/optimizer/net_optimizer_multi_head_attention.h"

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
#include "tnn/optimizer/graph_matcher/graph_registry.h"
#include "tnn/optimizer/graph_matcher/logger.h"

namespace TNN_NS {

namespace optimizer {

    std::shared_ptr<AnchorGraph> GetMatchedMHAAttentionMask(std::shared_ptr<Graph> graph, std::vector<std::shared_ptr<Graph>>& attn_mask_patterns,
                                                            NetResource *resource) {
        // Return Input Attention Mask module that matches.
        // Supported Module Kind right now:
        // 1: Huggingface 4+ , BERT Attention Mask Module
        // 2: TO BE ADDED
        // 3: ...

        // Step 1: Get Registered Patterns
        std::shared_ptr<Graph> huggingface_bert_attn_mask_pattern = attn_mask_patterns[0];

        // Step 2: Find Out Possible Modules using match() function.
        bool found_out_huggingface_bert_pattern = false;
        std::vector<std::shared_ptr<AnchorGraph>> matched_patterns;
        match(graph, huggingface_bert_attn_mask_pattern, matched_patterns);
        if (matched_patterns.size() == 1) {
            found_out_huggingface_bert_pattern = true;
        } else if (matched_patterns.size() > 1) {
            WARN("Found out Multiple Attention Mask Patterns in Graph, Ambiguous, Multi-Head Attenion Optimizer only applicable.");
            return nullptr;
        }

        // Step 3: Check Matched Module, return when all requirements meet.
        if (found_out_huggingface_bert_pattern) {
            std::shared_ptr<AnchorGraph> pattern = matched_patterns[0];

            auto slice0_node = pattern->getNodeByTensorName(std::string("@attn_mask_slice0"));
            if (!slice0_node || slice0_node->info->inputs.size() != 1) {
                WARN("Matched Huggingface Bert Slice0 Node Does Not Meet Requirements.");
                return nullptr;
            }
            auto slice0_param = dynamic_cast<StrideSliceV2LayerParam *>(slice0_node->info->param.get());
            if (!slice0_param ||
                slice0_param->begins.size() != 1 || slice0_param->ends.size() != 1 ||
                slice0_param->axes.size() != 1 || slice0_param->strides.size() != 1 ||
                slice0_param->begins[0] != 0 || slice0_param->ends[0] != INT_MAX ||
                slice0_param->axes[0] != 0 || slice0_param->strides[0] != 1) {
                WARN("Matched Huggingface Bert Slice0 Param does not meet requirements.");
                return nullptr;
            }

            auto unsqueeze0_node = pattern->getNodeByTensorName(std::string("@attn_mask_unsqueeze0"));
            if (!unsqueeze0_node || unsqueeze0_node->info->inputs.size() != 1) {
                WARN("Matched Huggingface Bert Unsqueeze0 Node Does Not Meet Requirements.");
                return nullptr;
            }
            auto unsqueeze0_param = dynamic_cast<UnsqueezeLayerParam *>(unsqueeze0_node->info->param.get());
            if (!unsqueeze0_param ||
                unsqueeze0_param->axes.size() != 1 || unsqueeze0_param->axes[0] != 1 ||
                unsqueeze0_param->data_in_resource != false) {
                WARN("Matched Huggingface Bert Unsqueeze0 Param does not meet requirements.");
                return nullptr;
            }

            auto unsqueeze1_node = pattern->getNodeByTensorName(std::string("@attn_mask_unsqueeze1"));
            if (!unsqueeze1_node || unsqueeze1_node->info->inputs.size() != 1) {
                WARN("Matched Huggingface Bert Unsqueeze0 Node Does Not Meet Requirements.");
                return nullptr;
            }
            auto unsqueeze1_param = dynamic_cast<UnsqueezeLayerParam *>(unsqueeze1_node->info->param.get());
            if (!unsqueeze1_param ||
                unsqueeze1_param->axes.size() != 1 || unsqueeze1_param->axes[0] != 2 ||
                unsqueeze1_param->data_in_resource != false) {
                WARN("Matched Huggingface Bert Unsqueeze1 Param does not meet requirements.");
                return nullptr;
            }

            auto slice1_node = pattern->getNodeByTensorName(std::string("@attn_mask_slice1"));
            if (!slice1_node || slice0_node->info->inputs.size() != 1) {
                WARN("Matched Huggingface Bert Slice1 Node Does Not Meet Requirements.");
                return nullptr;
            }
            auto slice1_param = dynamic_cast<StrideSliceV2LayerParam *>(slice1_node->info->param.get());
            if (!slice1_param ||
                slice1_param->begins.size() != 1 || slice1_param->ends.size() != 1 ||
                slice1_param->axes.size() != 1 || slice1_param->strides.size() != 1 ||
                slice1_param->begins[0] != 0 || slice1_param->ends[0] != INT_MAX ||
                slice1_param->axes[0] != 3 || slice1_param->strides[0] != 1) {
                WARN("Matched Huggingface Bert Slice1 Param does not meet requirements.");
                return nullptr;
            }

            auto cast_node = pattern->getNodeByTensorName(std::string("@attn_mask_cast_float"));
            if (!cast_node || cast_node->info->inputs.size() != 1) {
                WARN("Matched Huggingface Bert Type Cast to Float Node Does Not Meet Requirements.");
                return nullptr;
            }
            auto cast_param = dynamic_cast<CastLayerParam *>(cast_node->info->param.get());
            if (!cast_param || cast_param->to != 0) {
                WARN("Matched Huggingface Bert Type Cast to Float Param does not meet requirements.");
                return nullptr;
            }

            auto sub_node = pattern->getNodeByTensorName(std::string("@attn_mask_sub"));
            if (!sub_node || sub_node->info->inputs.size() != 1) {
                WARN("Matched Huggingface Bert Sub Node Does Not Meet Requirements.");
                return nullptr;
            }
            auto sub_param = dynamic_cast<MultidirBroadcastLayerParam *>(sub_node->info->param.get());
            if (!sub_param || sub_param->weight_input_index != 0) {
                WARN("Matched Huggingface Bert Sub Param does not meet requirements.");
                return nullptr;
            }
            auto sub_resource_map_iter = resource->resource_map.find(sub_node->info->name);
            if (sub_resource_map_iter == resource->resource_map.end()) {
                WARN("Unable to find out layer resource for Matched Huggingface Bert Sub Node in network constant map.");
                return nullptr;
            }
            auto sub_resource = dynamic_cast<EltwiseLayerResource *>(sub_resource_map_iter->second.get());
            if (!sub_resource || !(sub_resource->element_shape.size() == 0 ||
                                   (sub_resource->element_shape.size() == 1 && sub_resource->element_shape[0] == 1))) {
                WARN("Unable to get layer resource for Matched Huggingface Bert Sub Node in network resource map, bias.shape should be [1]");
                return nullptr;
            }
            if (sub_resource->element_handle.force_to<float*>()[0] != 1.0f) {
                WARN("Matched Huggingface Bert Sub Node, constant Value in position 0 should be 1.0f.");
                return nullptr;
            }

            auto mul_node = pattern->getNodeByTensorName(std::string("@attn_mask_mul"));
            if (!mul_node || mul_node->info->inputs.size() != 1) {
                WARN("Matched Huggingface Bert Mul Node Does Not Meet Requirements.");
                return nullptr;
            }
            auto mul_param = dynamic_cast<MultidirBroadcastLayerParam *>(mul_node->info->param.get());
            if (!mul_param || mul_param->weight_input_index != 1) {
                WARN("Matched Huggingface Bert Mul Param does not meet requirements.");
                return nullptr;
            }
            auto mul_resource_map_iter = resource->resource_map.find(mul_node->info->name);
            if (mul_resource_map_iter == resource->resource_map.end()) {
                WARN("Unable to find out layer resource for Matched Huggingface Bert Mul Node in network constant map.");
                return nullptr;
            }
            auto mul_resource = dynamic_cast<EltwiseLayerResource *>(mul_resource_map_iter->second.get());
            if (!mul_resource || !(mul_resource->element_shape.size() == 0 ||
                                   (mul_resource->element_shape.size() == 1 && mul_resource->element_shape[0] == 1))) {
                WARN("Unable to get layer resource for Matched Huggingface Bert Mul Node in network resource map, bias.shape should be [1]");
                return nullptr;
            }

            // All Checks Passed, Return Huggingface 4 Bert Attention Mask
            return pattern;
        }

        WARN("Unable to Find out Any Attention Mask Pattern in Graph, Multi-Head Attenion Optimizer only applicable.");
        return nullptr;
    }

    std::vector<std::shared_ptr<AnchorGraph>> GetMatchedMHAMainParts(std::shared_ptr<Graph> graph,
                                                                     std::vector<std::shared_ptr<Graph>>& mha_patterns, NetResource *resource,
                                                                     int& num_heads, int& per_head_size, int& hidden_size) {
        // Return Multi-Head Attention Main Part module that matches.
        // Supported Module Kind right now:
        // 1: Huggingface 4+ , BERT Multi-Head Attention Module
        // 2: TO BE ADDED
        // 3: ...

        // Shape List:
        // Hidden_size = Num_heads * Hidden_per_head
        //
        // in:                             [Batch, Seq_len, Hidden_size]
        // q/k/v_linear_mm/add:            [Batch, Seq_len, Hidden_size]
        // weight of q/k/v_linear_mm       [Hidden_size, Hidden_size]
        // bias of q/k/v_linear_add        [Hidden_size]
        // q/k/v_reshape:                  [Batch, Seq_len, Num_heads, Hidden_per_head]
        // q/k/v_permute:                  [Batch, Num_heads, Seq_len, Hidden_per_head]
        // k_permute_trans:                [Batch, Num_heads, Hidden_per_head, Seq_len]
        // attn_score/_div/_mask/_softmax: [Batch, Num_heads, Seq_len, Seq_len]
        // attn_context:                   [Batch, Num_heads, Seq_len, Hidden_per_head]
        // attn_context_permute:           [Batch, Seq_len, Num_heads, Hidden_per_head]
        // attn_context_reshape:           [Batch, Seq_len, Hidden_Size]
        
        // Step 1: get Registered Patterns

        // Parse Modules
        std::shared_ptr<Graph> huggingface_bert_mha_pattern = mha_patterns[0];


        // Step 2: Find Out Possible Modules using match() function.
        std::vector<std::shared_ptr<AnchorGraph>> matched_huggingface_bert_patterns;
        match(graph, huggingface_bert_mha_pattern, matched_huggingface_bert_patterns);
        if (!matched_huggingface_bert_patterns.empty() && matched_huggingface_bert_patterns.size() > 36) {
            // XXX Large Bert bigger than 36 layers are not supported.
            // 36 is added to limit cases when Large Amount of ambiguous matches are found.
            return {};
        }


        // Step 3: Find Out Possible Modules using match() function.
        std::vector<std::shared_ptr<AnchorGraph>> matched_patterns;
        //int num_heads     = -1;
        //int per_head_size = -1;
        //int hidden_size   = -1;
        for (int i=0; i<matched_huggingface_bert_patterns.size(); i++) {
            std::shared_ptr<AnchorGraph> pattern = matched_huggingface_bert_patterns[i];

            // Part 1:
            // Check QKV Reshape Concat Nodes
            auto q_reshape_concat_node  = pattern->getNodeByTensorName(std::string("@q_reshape_shape"));
            auto k_reshape_concat_node  = pattern->getNodeByTensorName(std::string("@k_reshape_shape"));
            auto v_reshape_concat_node  = pattern->getNodeByTensorName(std::string("@v_reshape_shape"));
            auto ac_reshape_concat_node = pattern->getNodeByTensorName(std::string("@ac_reshape_shape"));
            if (!q_reshape_concat_node || !k_reshape_concat_node || !v_reshape_concat_node || !ac_reshape_concat_node) {
                WARN("QKV & Attention Context Reshape Target Shape Concat Node not found in multi-head attention optimizer");
                return {};
            }
            if (q_reshape_concat_node->info->inputs.size() != 4 ||
                k_reshape_concat_node->info->inputs.size() != 4 ||
                v_reshape_concat_node->info->inputs.size() != 4) {
                WARN("QKV Reshape Target Shape Concat should have 4 inputs: [batch, seq_len, num_heads, hidden_size_per_head]");
                return {};
            }
            if (ac_reshape_concat_node->info->inputs.size() != 3) {
                WARN("Attention Context Reshape Target Shape Concat should have 3 inputs: [batch, seq_len, hidden_size]");
                return {};
            }
            std::string num_heads_node_name     = q_reshape_concat_node->info->inputs[2];
            std::string per_head_size_node_name = q_reshape_concat_node->info->inputs[3];
            std::string hidden_size_node_name   = ac_reshape_concat_node->info->inputs[2];
            if (num_heads_node_name != k_reshape_concat_node->info->inputs[2] ||
                num_heads_node_name != v_reshape_concat_node->info->inputs[2] ||
                per_head_size_node_name != k_reshape_concat_node->info->inputs[3] ||
                per_head_size_node_name != v_reshape_concat_node->info->inputs[3]) {
                WARN("num_heads, hidden_size_per_head of Q, K and V should be the same");
                return {};
            }
            auto num_heads_constant_map_iter     = resource->constant_map.find(num_heads_node_name);
            auto per_head_size_constant_map_iter = resource->constant_map.find(per_head_size_node_name);
            auto hidden_size_constant_map_iter   = resource->constant_map.find(hidden_size_node_name);
            if (num_heads_constant_map_iter == resource->constant_map.end() ||
                per_head_size_constant_map_iter == resource->constant_map.end() ||
                hidden_size_constant_map_iter == resource->constant_map.end()) {
                WARN("Unable to find out num_heads, hidden_size_per_head and hidden_size in network constant map.");
                return {};
            }

            const int curr_num_heads     = num_heads_constant_map_iter->second->force_to<int*>()[0];
            const int curr_per_head_size = per_head_size_constant_map_iter->second->force_to<int*>()[0];
            const int curr_hidden_size   = hidden_size_constant_map_iter->second->force_to<int*>()[0];
            num_heads                    = num_heads == -1 ? curr_num_heads : num_heads;
            per_head_size                = per_head_size == -1 ? curr_per_head_size : per_head_size;
            hidden_size                  = hidden_size == -1 ? curr_hidden_size : hidden_size;
            if (num_heads < 0 || per_head_size < 0 || hidden_size < 0 ||
                num_heads*per_head_size != hidden_size || num_heads != curr_num_heads ||
                per_head_size != curr_per_head_size || hidden_size != curr_hidden_size ||
                per_head_size != 64) {
                WARN("MHA Kernel num_heads * hidden_size_per_head should be equal to hidden_size, hidden_size, num_heads should be equal amongst all MHA modules, per_head_size should b 64.");
                return {};
            }


            // Part 2:
            // Check QKV Linear (matmul and add)
            auto q_linear_mm_node = pattern->getNodeByTensorName(std::string("@Module_Linear_0_qkv_linear_mm"));
            auto k_linear_mm_node = pattern->getNodeByTensorName(std::string("@Module_Linear_1_qkv_linear_mm"));
            auto v_linear_mm_node = pattern->getNodeByTensorName(std::string("@Module_Linear_2_qkv_linear_mm"));
            if (!q_linear_mm_node || !k_linear_mm_node || !v_linear_mm_node) {
                WARN("QKV Linear MatMul Node not found in multi-head attention optimizer");
                return {};
            }
            if (q_linear_mm_node->info->inputs.size() != 1 ||
                k_linear_mm_node->info->inputs.size() != 1 ||
                v_linear_mm_node->info->inputs.size() != 1 ||
                q_linear_mm_node->info->inputs[0] != k_linear_mm_node->info->inputs[0] ||
                q_linear_mm_node->info->inputs[0] != v_linear_mm_node->info->inputs[0]) {
                WARN("QKV Linear MatMul Node should have only one shared input.");
                return {};
            }
            auto q_linear_mm_param = dynamic_cast<MatMulLayerParam *>(q_linear_mm_node->info->param.get());
            auto k_linear_mm_param = dynamic_cast<MatMulLayerParam *>(k_linear_mm_node->info->param.get());
            auto v_linear_mm_param = dynamic_cast<MatMulLayerParam *>(v_linear_mm_node->info->param.get());
            if (!q_linear_mm_param || q_linear_mm_param->weight_position != 1 ||
                !k_linear_mm_param || k_linear_mm_param->weight_position != 1 ||
                !v_linear_mm_param || v_linear_mm_param->weight_position != 1) {
                WARN("QKV Linear MatMul Node param requires: param->weight_position == 1.");
                return {};
            }
            auto q_linear_mm_resource_map_iter = resource->resource_map.find(q_linear_mm_node->info->name);
            auto k_linear_mm_resource_map_iter = resource->resource_map.find(k_linear_mm_node->info->name);
            auto v_linear_mm_resource_map_iter = resource->resource_map.find(v_linear_mm_node->info->name);
            if (q_linear_mm_resource_map_iter == resource->resource_map.end() ||
                k_linear_mm_resource_map_iter == resource->resource_map.end() ||
                v_linear_mm_resource_map_iter == resource->resource_map.end()) {
                WARN("Unable to find out layer resource for Q, K and V Linear MatMul in network constant map.");
                return {};
            }
            auto q_linear_mm_resource = dynamic_cast<MatMulLayerResource *>(q_linear_mm_resource_map_iter->second.get());
            auto k_linear_mm_resource = dynamic_cast<MatMulLayerResource *>(k_linear_mm_resource_map_iter->second.get());
            auto v_linear_mm_resource = dynamic_cast<MatMulLayerResource *>(v_linear_mm_resource_map_iter->second.get());
            if (!q_linear_mm_resource || !k_linear_mm_resource || !v_linear_mm_resource) {
                WARN("Unable to get layer resource for Q, K and V Linear MatMul in network resource map.");
                return {};
            }
            if (q_linear_mm_resource->weight.GetBufferDims().size() != 2 ||
                k_linear_mm_resource->weight.GetBufferDims().size() != 2 ||
                v_linear_mm_resource->weight.GetBufferDims().size() != 2 ||
                q_linear_mm_resource->weight.GetBufferDims()[0] != hidden_size ||
                k_linear_mm_resource->weight.GetBufferDims()[0] != hidden_size ||
                v_linear_mm_resource->weight.GetBufferDims()[0] != hidden_size ||
                q_linear_mm_resource->weight.GetBufferDims()[1] != hidden_size ||
                k_linear_mm_resource->weight.GetBufferDims()[1] != hidden_size ||
                v_linear_mm_resource->weight.GetBufferDims()[1] != hidden_size) {
                WARN("QKV Linear MatMul Node weight requires shape: [hidden_size, hidden_size]");
                return {};
            }
            DataType qkv_linear_weight_bias_dtype = q_linear_mm_resource->weight.GetDataType();
            if (qkv_linear_weight_bias_dtype != k_linear_mm_resource->weight.GetDataType() ||
                qkv_linear_weight_bias_dtype != v_linear_mm_resource->weight.GetDataType() ||
                (qkv_linear_weight_bias_dtype != DATA_TYPE_FLOAT && qkv_linear_weight_bias_dtype != DATA_TYPE_HALF)) {
                WARN("DataType of weights of Q, K and V Linear in network resource map should be the same and should be Float or Half.");
                return {};
            }

            auto q_linear_add_node = pattern->getNodeByTensorName(std::string("@q_linear_add"));
            auto k_linear_add_node = pattern->getNodeByTensorName(std::string("@k_linear_add"));
            auto v_linear_add_node = pattern->getNodeByTensorName(std::string("@v_linear_add"));
            if (!q_linear_add_node || !k_linear_add_node || !v_linear_add_node) {
                WARN("QKV Linear Add Bias Node not found in multi-head attention optimizer");
                return {};
            }
            auto q_linear_add_param = dynamic_cast<MultidirBroadcastLayerParam *>(q_linear_add_node->info->param.get());
            auto k_linear_add_param = dynamic_cast<MultidirBroadcastLayerParam *>(k_linear_add_node->info->param.get());
            auto v_linear_add_param = dynamic_cast<MultidirBroadcastLayerParam *>(v_linear_add_node->info->param.get());
            if (q_linear_add_node->info->inputs.size() != 1 ||
                k_linear_add_node->info->inputs.size() != 1 ||
                v_linear_add_node->info->inputs.size() != 1 ||
                !q_linear_add_param || q_linear_add_param->weight_input_index != 1 ||
                !k_linear_add_param || k_linear_add_param->weight_input_index != 1 ||
                !v_linear_add_param || v_linear_add_param->weight_input_index != 1) {
                WARN("QKV Linear Add Bias Node should have only one input node with param->weight_input_index == 1");
                return {};
            }
            auto q_linear_add_resource_map_iter = resource->resource_map.find(q_linear_add_node->info->name);
            auto k_linear_add_resource_map_iter = resource->resource_map.find(k_linear_add_node->info->name);
            auto v_linear_add_resource_map_iter = resource->resource_map.find(v_linear_add_node->info->name);
            if (q_linear_add_resource_map_iter == resource->resource_map.end() ||
                k_linear_add_resource_map_iter == resource->resource_map.end() ||
                v_linear_add_resource_map_iter == resource->resource_map.end()) {
                WARN("Unable to find out layer resource for Q, K and V Linear Bias Add in network constant map.");
                return {};
            }
            auto q_linear_add_resource = dynamic_cast<EltwiseLayerResource *>(q_linear_add_resource_map_iter->second.get());
            auto k_linear_add_resource = dynamic_cast<EltwiseLayerResource *>(k_linear_add_resource_map_iter->second.get());
            auto v_linear_add_resource = dynamic_cast<EltwiseLayerResource *>(v_linear_add_resource_map_iter->second.get());
            if (!q_linear_add_resource || !k_linear_add_resource || !v_linear_add_resource ||
                q_linear_add_resource->element_handle.GetBufferDims().size() != 1 ||
                k_linear_add_resource->element_handle.GetBufferDims().size() != 1 ||
                v_linear_add_resource->element_handle.GetBufferDims().size() != 1 ||
                q_linear_add_resource->element_handle.GetBufferDims()[0] != hidden_size ||
                k_linear_add_resource->element_handle.GetBufferDims()[0] != hidden_size ||
                v_linear_add_resource->element_handle.GetBufferDims()[0] != hidden_size) {
                WARN("Unable to get layer resource for Q, K and V Linear Bias Add in network resource map, bias.shape should be [hidden_size]");
                return {};
            }
            if (qkv_linear_weight_bias_dtype != q_linear_add_resource->element_handle.GetDataType() ||
                qkv_linear_weight_bias_dtype != k_linear_add_resource->element_handle.GetDataType() ||
                qkv_linear_weight_bias_dtype != v_linear_add_resource->element_handle.GetDataType()) {
                WARN("DataType of weights of Q, K and V Linear in network resource map should be the same as weight dtype and should be Float or Half.");
                return {};
            }

            // Everything All-right,
            matched_patterns.emplace_back(pattern);
        }
        return matched_patterns;

        WARN("Unable to Find out Any Multi-Head Attention Pattern in Graph, Multi-Head Attenion Optimizer only applicable.");
        return {};
    }

    bool CheckAttnMaskMainPartConnection(std::shared_ptr<AnchorGraph> matched_attn_mask_pattern, std::vector<std::shared_ptr<AnchorGraph>>& matched_mha_patterns) {
        auto attn_mask_mul_node = matched_attn_mask_pattern->getNodeByTensorName(std::string("@attn_mask_mul"));
        std::string mask_out_name = attn_mask_mul_node->info->outputs[0];

        for (auto mha_pattern : matched_mha_patterns) {
            auto attn_score_mask_node = mha_pattern->getNodeByTensorName(std::string("@attn_score_mask"));
            std::string mha_in_name = attn_score_mask_node->info->inputs[1];
            if (mha_in_name != mask_out_name) {
                WARN("Attention Mask Out And Multi-Head Attention Input not connected.");
                return false;
            }
        }
        return true;
    }


    bool rewriteAttentionMask(std::shared_ptr<Graph> graph, std::shared_ptr<AnchorGraph> matched_pattern) {
        // Use 'embed()' to update ond graph pattern with 1 output with new cumsum output.

        // Input Attention Mask: [Batch, Sequence Length]
        // Cast to Int32:        [Batch, Sequence Length]
        // Get Padding Idx:      [Batch+1]  for dense mode, [2*batch+1] for sparse mode
        auto subgraph_to_replace = std::make_shared<TNN_NS::Graph>();

        std::string in_attn_mask_name         = "in_attn_mask";
        std::string attn_mask_int32_name      = "opt_mha_attn_mask_int32";
        std::string padding_idx_name          = "opt_mha_padding_idx";
        auto in_attn_mask_node                = subgraph_to_replace->getNodeOrCreatePlaceHolder(in_attn_mask_name);

        // Step 1: Cast to Int32
        if (subgraph_to_replace->createNode(LAYER_CAST, {in_attn_mask_name}, {attn_mask_int32_name}) != TNN_OK) {
            WARN("Unable to create new node for NEW MHA reduced attention_mask type cast to float in TNN Graph Mather.");
            return false;
        }
        auto new_mask_cast_int32_node         = subgraph_to_replace->getNodeByTensorName(attn_mask_int32_name);
        new_mask_cast_int32_node->info->param = std::make_shared<CastLayerParam>();
        auto new_mask_cast_int32_layer_param  = dynamic_cast<CastLayerParam *>(new_mask_cast_int32_node->info->param.get());
        if (!new_mask_cast_int32_layer_param) {
            WARN("Unable to initialize CastLayerParam for NEW MHA reduced attention_mask type cast to INT32 node.");
            return false;
        }
        new_mask_cast_int32_layer_param->to   = DATA_TYPE_INT32;

        // Step 2: Get Padding Index (cu_seqlen) for QKVtoContext,
        //         For Sparse Mode: [Batch, Sequence Length] -> [2*Batch+1]
        //         For Dense Mode:  [Batch, Sequence Length] -> [Batch+1]
        if (subgraph_to_replace->createNode(LAYER_EFFECTIVE_TRANSFORMER, {attn_mask_int32_name}, {padding_idx_name}) != TNN_OK) {
            WARN("Unable to create new node for NEW MHA Padding index from Attention Mask Node in TNN Graph Mather.");
            return false;
        }
        auto new_padding_idx_node         = subgraph_to_replace->getNodeByTensorName(padding_idx_name);
        new_padding_idx_node->info->param = std::make_shared<EffectiveTransformerLayerParam>();
        auto new_padding_idx_layer_param  = dynamic_cast<EffectiveTransformerLayerParam *>(new_padding_idx_node->info->param.get());
        if (!new_padding_idx_layer_param) {
            WARN("Unable to initialize ReduceLayerParam for NEW MHA Padding index from Attention Mask Node .");
            return false;
        }
        new_padding_idx_layer_param->is_remove_padding = false;   // Sparse Mode
        //new_padding_idx_layer_param->is_remove_padding = true;    // Dense Mode
        // new_padding_idx_layer_param->get_padding_offset_from_2d_mask = true;

        // Step 3: Run embed() to add to graph.
        subgraph_to_replace->embed(graph->shared_from_this(), matched_pattern, "");

        return true;
    }

    bool rewriteMultiHeadAttention(std::shared_ptr<Graph> graph, std::vector<std::shared_ptr<Graph>> mha_patterns_vec, NetResource *resource,
                                   const int num_heads, const int per_head_size, const int hidden_size) {
        // Global Steps:
        // Global Step 1: Re-Match Multi-Head Attention Modules with attention mask input replaced by cumsum.
        // Global Step 2: Run Loops to Create New Nodes, new resources of Optimized MHA kernels, rewrite graph.
        //                In Step 2, we call embed(), Plugin: QKVtoContext 'max_seqlen' input is not ready at this time.
        // Global Step 3: Add 'max_seqlen' generation node to graph, replace all Optimized MHA QKVtoContext input 3 with this 'max_seqlen'


        // Global Step 1:
        //                Re-Match Multi-Head Attention Modules with attention mask input replaced by cumsum.
        // In Previous Functions, 'attention mask' in graph has been replaced with attention_mask cumsum,
        // We have to call match here once again.
        std::shared_ptr<Graph> huggingface_bert_mha_pattern = mha_patterns_vec[0];
        std::vector<std::shared_ptr<AnchorGraph>> matched_huggingface_bert_patterns;
        match(graph, huggingface_bert_mha_pattern, matched_huggingface_bert_patterns);
        if (!matched_huggingface_bert_patterns.empty() && matched_huggingface_bert_patterns.size() > 36) {
            // XXX Large Bert bigger than 36 layers are not supported.
            // 36 is added to limit cases when Large Amount of ambiguous matches are found.
            return false;
        }


        
        // Global Step 2:
        //                Run Loops to Create New Nodes, new resources of Optimized MHA kernels, rewrite graph.
        //                In Step 2, we call embed(), Plugin: QKVtoContext 'max_seqlen' input is not ready at this time.
        //                So we will record all new QKVtoContext Nodes to be created in the following code piece.
        //                Their inputs 'max_seqlen' would be created and replaced in the next step.
        std::vector<std::shared_ptr<Node>> new_qkv_to_context_nodes_vec;
        for (int i=0; i<matched_huggingface_bert_patterns.size(); i++) {
            std::shared_ptr<AnchorGraph> matched_pattern = matched_huggingface_bert_patterns[i];

            auto subgraph_to_replace = std::make_shared<TNN_NS::Graph>();

            auto in_x_name                = "in_x";
            auto in_attn_padding_idx_name = "opt_mha_attn_mask_cumsum";    // Cumsum of Attention Mask Already been replace by cumsum in this step.
            auto in_num_heads_name        = "in_num_heads";
            auto in_per_head_size_name    = "in_per_head_size";
            auto in_hidden_size_name      = "in_hidden_size";
            auto in_x_node                = subgraph_to_replace->getNodeOrCreatePlaceHolder(in_x_name);
            auto in_attn_padding_idx_node = subgraph_to_replace->getNodeOrCreatePlaceHolder(in_attn_padding_idx_name);
            auto in_num_heads_node        = subgraph_to_replace->getNodeOrCreatePlaceHolder(in_num_heads_name);
            auto in_per_head_size_node    = subgraph_to_replace->getNodeOrCreatePlaceHolder(in_per_head_size_name);
            auto in_hidden_size_node      = subgraph_to_replace->getNodeOrCreatePlaceHolder(in_hidden_size_name);

            std::string in_x_shape_name               = "opt_mha_in_shape_" + std::to_string(i);
            std::string merged_qkv_linear_mm_name     = "opt_mha_qkv_linear_matmul_" + std::to_string(i);
            std::string merged_qkv_linear_add_name    = "opt_mha_qkv_linear_add_" + std::to_string(i);
            std::string mha_reshape_to_2d_name        = "opt_mha_reshape_to_2d_" + std::to_string(i);
            std::string mha_unsqueeze_name            = "opt_mha_unsqueeze_" + std::to_string(i);
            std::string mha_fused_qkv_to_context_name = "opt_mha_fused_qkv_to_context" + std::to_string(i);
            std::string mha_squeeze_name              = "opt_mha_squeeze_" + std::to_string(i);
            std::string mha_squeeze_reshape_name      = "opt_mha_squeeze_reshape_" + std::to_string(i);


            // Step 1: Get Shape of in_x
            //         Shape will be used after TensorRT QKVToContext V2 Plugin to reshape X back to its origin shape
            if (subgraph_to_replace->createNode(LAYER_SHAPE, {in_x_name}, {in_x_shape_name}) != TNN_OK) {
                WARN("Unable to create new node for NEW MHA input shape in TNN Graph Mather.");
                return false;
            }
            auto new_input_shape_node         = subgraph_to_replace->getNodeByTensorName(in_x_shape_name);
            new_input_shape_node->info->param = std::make_shared<LayerParam>();


            // Step 2: Combine 'input to QKV' linear from Q,K,V separete 3 into merged 1.
            //         Reorder weight of merged linear.
            //         from    [Batch, Seq_len, Hidden_size] to [Batch, Seq_len, 3*Hidden_size] in sparse mode.
            //         or from [Batch*Seq_len,  Hidden_size] to [Batch*Seq_len,  3*Hidden_size] in dense mode.
            // RawBuffer will call malloc and memcpy in its initialization.
            auto q_linear_mm_node = matched_pattern->getNodeByTensorName(std::string("@Module_Linear_0_qkv_linear_mm"));
            auto k_linear_mm_node = matched_pattern->getNodeByTensorName(std::string("@Module_Linear_1_qkv_linear_mm"));
            auto v_linear_mm_node = matched_pattern->getNodeByTensorName(std::string("@Module_Linear_2_qkv_linear_mm"));
            if (!q_linear_mm_node || !k_linear_mm_node || !v_linear_mm_node) {
                WARN("QKV Linear MatMul Node not found in multi-head attention pattern");
                return false;
            }
            auto q_linear_mm_resource_map_iter = resource->resource_map.find(q_linear_mm_node->info->name);
            auto k_linear_mm_resource_map_iter = resource->resource_map.find(k_linear_mm_node->info->name);
            auto v_linear_mm_resource_map_iter = resource->resource_map.find(v_linear_mm_node->info->name);
            auto q_linear_mm_resource = dynamic_cast<MatMulLayerResource *>(q_linear_mm_resource_map_iter->second.get());
            auto k_linear_mm_resource = dynamic_cast<MatMulLayerResource *>(k_linear_mm_resource_map_iter->second.get());
            auto v_linear_mm_resource = dynamic_cast<MatMulLayerResource *>(v_linear_mm_resource_map_iter->second.get());

            DataType qkv_linear_weight_bias_dtype = q_linear_mm_resource->weight.GetDataType();
            int qkv_linear_dtype_size             = qkv_linear_weight_bias_dtype==DATA_TYPE_FLOAT ? 4 : 2;
            char* tmp_qkv_linear_mm_ptr        = (char*)std::malloc(qkv_linear_dtype_size * hidden_size * 3*hidden_size);
            char* q_linear_mm_weight_buf_ptr   = q_linear_mm_resource->weight.force_to<char*>();
            char* k_linear_mm_weight_buf_ptr   = k_linear_mm_resource->weight.force_to<char*>();
            char* v_linear_mm_weight_buf_ptr   = v_linear_mm_resource->weight.force_to<char*>();
            for (int i=0; i<hidden_size; i++) {
                for (int h=0; h<num_heads; h++) {
                    std::memcpy(tmp_qkv_linear_mm_ptr + (i*3*hidden_size + (3*h+0)*per_head_size)*qkv_linear_dtype_size,
                                q_linear_mm_weight_buf_ptr + (i*hidden_size + h*per_head_size)*qkv_linear_dtype_size,
                                qkv_linear_dtype_size*per_head_size);
                    std::memcpy(tmp_qkv_linear_mm_ptr + (i*3*hidden_size + (3*h+1)*per_head_size)*qkv_linear_dtype_size,
                                k_linear_mm_weight_buf_ptr + (i*hidden_size + h*per_head_size)*qkv_linear_dtype_size,
                                qkv_linear_dtype_size*per_head_size);
                    std::memcpy(tmp_qkv_linear_mm_ptr + (i*3*hidden_size + (3*h+2)*per_head_size)*qkv_linear_dtype_size,
                                v_linear_mm_weight_buf_ptr + (i*hidden_size + h*per_head_size)*qkv_linear_dtype_size,
                                qkv_linear_dtype_size*per_head_size);
                }
            }
            RawBuffer merged_qkv_linear_mm_buf = RawBuffer(hidden_size * 3*hidden_size * qkv_linear_dtype_size,
                                                           tmp_qkv_linear_mm_ptr, {hidden_size, 3*hidden_size});
            merged_qkv_linear_mm_buf.SetDataType(qkv_linear_weight_bias_dtype);
            std::free(tmp_qkv_linear_mm_ptr);
            auto merged_qkv_linear_mm_res      = new MatMulLayerResource();
            merged_qkv_linear_mm_res->weight   = merged_qkv_linear_mm_buf;

            // RawBuffer will call malloc and memcpy in its initialization.
            auto q_linear_add_node = matched_pattern->getNodeByTensorName(std::string("@q_linear_add"));
            auto k_linear_add_node = matched_pattern->getNodeByTensorName(std::string("@k_linear_add"));
            auto v_linear_add_node = matched_pattern->getNodeByTensorName(std::string("@v_linear_add"));
            if (!q_linear_add_node || !k_linear_add_node || !v_linear_add_node) {
                WARN("QKV Linear Add Bias Node not found in multi-head attention optimizer");
                return false;
            }
            auto q_linear_add_resource_map_iter = resource->resource_map.find(q_linear_add_node->info->name);
            auto k_linear_add_resource_map_iter = resource->resource_map.find(k_linear_add_node->info->name);
            auto v_linear_add_resource_map_iter = resource->resource_map.find(v_linear_add_node->info->name);
            auto q_linear_add_resource = dynamic_cast<EltwiseLayerResource *>(q_linear_add_resource_map_iter->second.get());
            auto k_linear_add_resource = dynamic_cast<EltwiseLayerResource *>(k_linear_add_resource_map_iter->second.get());
            auto v_linear_add_resource = dynamic_cast<EltwiseLayerResource *>(v_linear_add_resource_map_iter->second.get());

            char* tmp_qkv_linear_add_ptr    = (char*)std::malloc(qkv_linear_dtype_size * 3*hidden_size);
            char* q_linear_add_bias_buf_ptr = q_linear_add_resource->element_handle.force_to<char*>();
            char* k_linear_add_bias_buf_ptr = k_linear_add_resource->element_handle.force_to<char*>();
            char* v_linear_add_bias_buf_ptr = v_linear_add_resource->element_handle.force_to<char*>();
            for (int h=0; h<num_heads; h++) {
                std::memcpy(tmp_qkv_linear_add_ptr + (3*h+0)*per_head_size*qkv_linear_dtype_size,
                            q_linear_add_bias_buf_ptr + h*per_head_size*qkv_linear_dtype_size,
                            qkv_linear_dtype_size*per_head_size);
                std::memcpy(tmp_qkv_linear_add_ptr + (3*h+1)*per_head_size*qkv_linear_dtype_size,
                            k_linear_add_bias_buf_ptr + h*per_head_size*qkv_linear_dtype_size,
                            qkv_linear_dtype_size*per_head_size);
                std::memcpy(tmp_qkv_linear_add_ptr + (3*h+2)*per_head_size*qkv_linear_dtype_size,
                            v_linear_add_bias_buf_ptr + h*per_head_size*qkv_linear_dtype_size,
                            qkv_linear_dtype_size*per_head_size);
            }
            RawBuffer merged_qkv_linear_add_buf = RawBuffer(3*hidden_size * qkv_linear_dtype_size,
                                                            tmp_qkv_linear_add_ptr, {3*hidden_size});
            merged_qkv_linear_add_buf.SetDataType(qkv_linear_weight_bias_dtype);
            std::free(tmp_qkv_linear_add_ptr);
            auto merged_qkv_linear_add_res            = new EltwiseLayerResource();
            merged_qkv_linear_add_res->element_handle = merged_qkv_linear_add_buf;
            merged_qkv_linear_add_res->element_shape  = merged_qkv_linear_add_buf.GetBufferDims();
            resource->resource_map[merged_qkv_linear_mm_name]  = std::shared_ptr<LayerResource>(merged_qkv_linear_mm_res);
            resource->resource_map[merged_qkv_linear_add_name] = std::shared_ptr<LayerResource>(merged_qkv_linear_add_res);

            // Insert New Merged Linear Node (MatMul Part)
            if (subgraph_to_replace->createNode(LAYER_MATMUL, {in_x_name}, {merged_qkv_linear_mm_name}) != TNN_OK) {
                WARN("Unable to create new node for NEW MHA merged QKV MatMul in TNN Graph Mather.");
                return false;
            }
            auto new_merged_qkv_linear_mm_node         = subgraph_to_replace->getNodeByTensorName(merged_qkv_linear_mm_name);
            new_merged_qkv_linear_mm_node->info->param = std::make_shared<MatMulLayerParam>();
            auto new_merged_qkv_linear_mm_layer_param  = dynamic_cast<MatMulLayerParam *>(new_merged_qkv_linear_mm_node->info->param.get());
            if (!new_merged_qkv_linear_mm_layer_param) {
                WARN("Unable to initialize MatMulLayerParam for NEW MHA merged QKV MatMul node.");
                return false;
            }
            new_merged_qkv_linear_mm_layer_param->weight_position       = 1;
            new_merged_qkv_linear_mm_layer_param->matrix_b_dims         = {hidden_size, 3*hidden_size};

            // Insert New Merged Linear Node (Add Part)
            if (subgraph_to_replace->createNode(LAYER_ADD, {merged_qkv_linear_mm_name}, {merged_qkv_linear_add_name}) != TNN_OK) {
                WARN("Unable to create new node for NEW MHA merged QKV Add in TNN Graph Mather.");
                return false;
            }
            auto new_merged_qkv_linear_add_node         = subgraph_to_replace->getNodeByTensorName(merged_qkv_linear_add_name);
            new_merged_qkv_linear_add_node->info->param = std::make_shared<MultidirBroadcastLayerParam>();
            auto new_merged_qkv_linear_add_layer_param  = dynamic_cast<MultidirBroadcastLayerParam *>(new_merged_qkv_linear_add_node->info->param.get());
            if (!new_merged_qkv_linear_add_layer_param) {
                WARN("Unable to initialize MatMulLayerParam for NEW MHA merged QKV Add node.");
                return false;
            }
            new_merged_qkv_linear_add_layer_param->weight_input_index = 1;


            // Step 3: Merge dim0 and dim1 for QKVtoContext V2 Var SeqLen Mode.
            //         from [Batch, Seq_len, 3*Hidden_size] to [Batch*Seq_len, 3*Hidden_size] in spase mode.
            //         or shape is not changed, remain to be [Batch*Seq_len, 3*Hidden_size] in dense mode.
            if (subgraph_to_replace->createNode(LAYER_RESHAPE, {merged_qkv_linear_add_name}, {mha_reshape_to_2d_name}) != TNN_OK) {
                WARN("Unable to create new node for NEW MHA unsqueeze for Plugin in TNN Graph Mather.");
                return false;
            }
            auto new_mha_reshape_to_2d_node             = subgraph_to_replace->getNodeByTensorName(mha_reshape_to_2d_name);
            new_mha_reshape_to_2d_node->info->param     = std::make_shared<ReshapeLayerParam>();
            auto new_mha_reshape_to_2d_layer_param      = dynamic_cast<ReshapeLayerParam *>(new_mha_reshape_to_2d_node->info->param.get());
            if (!new_mha_reshape_to_2d_layer_param) {
                WARN("Unable to initialize UnsqueezeLayerParam for NEW MHA Unsqueeze node.");
                return false;
            }
            new_mha_reshape_to_2d_layer_param->shape    = {-1, 3*hidden_size};


            // Step 4: Add two Unsqueeze after merged 'input to QKV' linear.
            //         from [Batch*Seq_len, 3*Hidden_size] to [Batch*Seq_len, 3*Hidden_size， 1， 1]
            if (subgraph_to_replace->createNode(LAYER_UNSQUEEZE, {mha_reshape_to_2d_name}, {mha_unsqueeze_name}) != TNN_OK) {
                WARN("Unable to create new node for NEW MHA unsqueeze for Plugin in TNN Graph Mather.");
                return false;
            }
            auto new_mha_unsqueeze_node         = subgraph_to_replace->getNodeByTensorName(mha_unsqueeze_name);
            new_mha_unsqueeze_node->info->param = std::make_shared<UnsqueezeLayerParam>();
            auto new_mha_unsqueeze_layer_param  = dynamic_cast<UnsqueezeLayerParam *>(new_mha_unsqueeze_node->info->param.get());
            if (!new_mha_unsqueeze_layer_param) {
                WARN("Unable to initialize UnsqueezeLayerParam for NEW MHA unsqueeze for Plugin node.");
                return false;
            }
            new_mha_unsqueeze_layer_param->axes = {2, 3};


            // Step 5: Add QKVtoContext V3 TRT Plugin
            //         from [Batch*Seq_len, 3*Hidden_size， 1， 1] to [Batch*Seq_len, Hidden_size, 1, 1]
            // In this step, input3 of MHA node is qkv linear add, which is not correct
            // it would be replace with 'max_seqlen' in the next steps.
            if (subgraph_to_replace->createNode(LAYER_FUSED, {mha_unsqueeze_name, merged_qkv_linear_mm_name, in_attn_padding_idx_name, merged_qkv_linear_add_name}, {mha_fused_qkv_to_context_name}) != TNN_OK) {
                WARN("Unable to create new node for NEW MHA merged QKV to Context V2 in TNN Graph Mather.");
                return false;
            }
            auto new_mha_fused_qkv_to_context_node         = subgraph_to_replace->getNodeByTensorName(mha_fused_qkv_to_context_name);
            new_mha_fused_qkv_to_context_node->info->param = std::make_shared<FusedLayerParam>();
            auto new_mha_fused_qkv_to_context_layer_param  = dynamic_cast<FusedLayerParam *>(new_mha_fused_qkv_to_context_node->info->param.get());
            if (!new_mha_fused_qkv_to_context_layer_param) {
                WARN("Unable to initialize FusedLayerParam for NEW MHA merged QKV to Context V2 node.");
                return false;
            }
            new_mha_fused_qkv_to_context_layer_param->type                 = FusionType_TRTPlugin_BertQKVtoContextV2;
            new_mha_fused_qkv_to_context_layer_param->bert_mha_hidden_size = hidden_size;
            new_mha_fused_qkv_to_context_layer_param->bert_mha_num_heads   = num_heads;
            // Store Nodes, replace input 3 with 'max_seqlen' in the next steps.
            new_qkv_to_context_nodes_vec.push_back(new_mha_fused_qkv_to_context_node);


            // Step 6: Add two Squeeze after merged QKVtoContext V3 TRT Plugin
            //         from [Batch*Seq_len, Hidden_size, 1, 1] to [Batch*Seq_len, Hidden_size]
            if (subgraph_to_replace->createNode(LAYER_SQUEEZE, {mha_fused_qkv_to_context_name}, {mha_squeeze_name}) != TNN_OK) {
                WARN("Unable to create new node for NEW MHA Squeeze in TNN Graph Mather.");
                return false;
            }
            auto new_mha_squeeze_node         = subgraph_to_replace->getNodeByTensorName(mha_squeeze_name);
            new_mha_squeeze_node->info->param = std::make_shared<SqueezeLayerParam>();
            auto new_mha_squeeze_layer_param  = dynamic_cast<SqueezeLayerParam *>(new_mha_squeeze_node->info->param.get());
            if (!new_mha_squeeze_layer_param) {
                WARN("Unable to initialize SqueezeLayerParam for NEW MHA Squeeze node.");
                return false;
            }
            new_mha_squeeze_layer_param->axes = {2, 3};


            // Step 7: Reshape Back, split Batch, Seq_len
            //         from [Batch*Seq_len, Hidden_size] to [Batch, Seq_len, Hidden_size] under sparse mode.
            //         or keep origin shape [Batch*Seq_len, Hidden_size] under dense mode.
            if (subgraph_to_replace->createNode(LAYER_RESHAPE, {mha_squeeze_name, in_x_shape_name}, {mha_squeeze_reshape_name}) != TNN_OK) {
                WARN("Unable to create new node for NEW MHA squeeze reshape in TNN Graph Mather.");
                return false;
            }
            auto new_mha_squeeze_reshape_node         = subgraph_to_replace->getNodeByTensorName(mha_squeeze_reshape_name);
            new_mha_squeeze_reshape_node->info->param = std::make_shared<ReshapeLayerParam>();
            auto new_mha_squeeze_reshape_layer_param  = dynamic_cast<ReshapeLayerParam *>(new_mha_squeeze_reshape_node->info->param.get());
            if (!new_mha_squeeze_reshape_layer_param) {
                WARN("Unable to initialize ReshapeLayerParam for NEW MHA squeeze reshape node.");
                return false;
            }
            new_mha_squeeze_reshape_layer_param->num_axes = 0;


            // Step 8: Run embed() to add to graph.
            subgraph_to_replace->embed(graph->shared_from_this(), matched_pattern, "");
        }



        // Global Step 3:
        //                Add 'max_seqlen' generation node to graph, replace all Optimized MHA QKVtoContext input 3 with this 'max_seqlen'
        // 3.1: Get input attention mask node from matched attention mask pattern.
        if (new_qkv_to_context_nodes_vec.empty()) {
            WARN("Unable to get Optimized new QKV to Context node.");
            return false;
        }

        // Get Input Attention Mask Node from QKVToContext Node through their topology relation chains.
        std::shared_ptr<Node> qkv_to_context_node0 = new_qkv_to_context_nodes_vec[0];
        if (qkv_to_context_node0->input_edges.size() != 4) {
            WARN("Optimized new QKV to Context node should have 4 inputs.");
            return false;
        }

        Edge* attn_mask_pad_idx_to_plugin_edge = qkv_to_context_node0->input_edges[2];
        if (attn_mask_pad_idx_to_plugin_edge == nullptr) {
            WARN("Unable to get Edge from attention mask Padding index to QKVtoContext Plugin.");
            return false;
        }
        Node* attn_mask_pad_idx_node = attn_mask_pad_idx_to_plugin_edge->src;
        if (attn_mask_pad_idx_node == nullptr) {
            WARN("Unable to get Attention Mask Cumsum Node.");
            return false;
        }

        Edge* attn_mask_cast_to_pad_idx_edge = attn_mask_pad_idx_node->input_edges[0];
        if (attn_mask_cast_to_pad_idx_edge == nullptr) {
            WARN("Unable to get Edge from attention mask input to pad idx Plugin.");
            return false;
        }
        Node* attn_mask_input_cast_node = attn_mask_cast_to_pad_idx_edge->src;
        if (attn_mask_input_cast_node == nullptr) {
            WARN("Unable to get Attention Mask Input Cast to Int32 Node.");
            return false;
        }

        // 3.2: Add Reduce Sum Node to get Max Sequence Length
        std::string attn_mask_max_seqlen_name = "opt_mha_attn_mask_max_seqlen";
        if (graph->createNode(LAYER_REDUCE_SUM, {attn_mask_input_cast_node->name()}, {attn_mask_max_seqlen_name}) != TNN_OK) {
            WARN("Unable to create new node for NEW MHA QKV to Context V2 Gen max seqlen in TNN Graph Matcher.");
            return false;
        }
        auto new_attn_mask_max_seqlen_node         = graph->getNodeByTensorName(attn_mask_max_seqlen_name);
        new_attn_mask_max_seqlen_node->info->param = std::make_shared<ReduceLayerParam>();
        auto new_attn_mask_max_seqlen_layer_param  = dynamic_cast<ReduceLayerParam *>(new_attn_mask_max_seqlen_node->info->param.get());
        if (!new_attn_mask_max_seqlen_layer_param) {
            WARN("Unable to initialize ReduceLayerParam for NEW MHA QKV to Context V2 Gen max seqlen node.");
            return false;
        }
        new_attn_mask_max_seqlen_layer_param->axis = {0};
        new_attn_mask_max_seqlen_layer_param->keep_dims = false;  // Plugin V2, out is [Seq Len]

        // 3.3: Update input[3] of QKVtoContext Nodes
        for (auto qkv_to_context_node : new_qkv_to_context_nodes_vec) {
            if (qkv_to_context_node->input_edges.size() != 4) {
                WARN("Optimized new QKV to Context node should have 4 inputs.");
                return false;
            }

            Edge* max_seqlen_edge = qkv_to_context_node->input_edges[3];
            if (max_seqlen_edge == nullptr) {
                WARN("Unable to get Edge from max sequence length to QKVtoContext.");
                return false;
            }
            Node* old_placeholder_node = max_seqlen_edge->src;
            if (old_placeholder_node == nullptr) {
                WARN("Unable to get old Padded Input Node.");
                return false;
            }

            auto status = qkv_to_context_node->updateInput(old_placeholder_node->name(), attn_mask_max_seqlen_name, max_seqlen_edge);
            if (status != TNN_OK) {
                WARN("Unable to update input3 of QKVtoContext Plugin from old Padded Input Node to new Max Sequence Length Node.");
                return false;
            }
        }

        // 3.4: Update Graph, add New Max SeqLen Node to Graph
        graph->reorderNodeAfter(attn_mask_input_cast_node->name(), attn_mask_max_seqlen_name);
        graph->updateTnnNetStructure();
        
        return true;
    }




    NetOptimizerRegister<NetOptimizerMultiHeadAttention> g_net_optimizer_multi_head_attn(OptPriority::P1);

    std::string NetOptimizerMultiHeadAttention::Strategy() {
        return kNetOptimizerMultiHeadAttention;
    }

    bool NetOptimizerMultiHeadAttention::IsSupported(const NetworkConfig &net_config) {
        return false;

        auto device = net_config.device_type;
        if (device == DEVICE_CUDA) {
            /*
            // TODO: CHECK TRT Version
            // SHOULD FIND a way to include <NvInfer.h> first
#ifndef NV_TENSORRT_MAJOR
            std::cout << "============= Net Optimizer MHA, TRT VERSION NOT AVAILABLE HERE !!!! =============" << std::endl;
#else
            int trt_version_num = NV_TENSORRT_MAJOR * 100 + NV_TENSORRT_MINOR * 10 + NV_TENSORRT_PATCH;
            std::cout << "============= Net Optimizer MHA, TRT VERSION = " << trt_version_num << " =============" << std::endl;
            if (trt_version_num < 824) {
                return false;
            }
#endif
            */
            return true;
        }
        return false;
    }

    Status NetOptimizerMultiHeadAttention::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::shared_ptr<Graph> graph = std::make_shared<Graph>();
        RETURN_ON_FAIL(graph->fromInterpreted(structure, resource));


        // Global Optimize Steps:
        // [1]. List Patterns && Modules
        //      Graph will not be rewrited in this step.
        TNN_NS::GraphRegistry registry;
        TNN_NS::GraphParser graph_parser(&registry);

        // 1.1: Attention Mask Ptterns
        std::vector<std::shared_ptr<Graph>> attn_mask_patterns;

        std::string pattern_huggingface_bert_attn_mask_str = R"(
            graph(%in_attn_mask):
                %attn_mask_slice0         = StridedSliceV2(%in_attn_mask)
                %attn_mask_unsqueeze0     = Unsqueeze(%attn_mask_slice0)
                %attn_mask_unsqueeze1     = Unsqueeze(%attn_mask_unsqueeze0)
                %attn_mask_slice1         = StridedSliceV2(%attn_mask_unsqueeze1)
                %attn_mask_cast_float     = Cast(%attn_mask_slice1)
                %attn_mask_sub            = Sub(%attn_mask_cast_float)
                %attn_mask_mul            = Mul(%attn_mask_sub)
            return (%attn_mask_mul)
        )";
        if (graph_parser.parseFromString(pattern_huggingface_bert_attn_mask_str)) {
            attn_mask_patterns.emplace_back(graph_parser.getGraph());
        } else {
            WARN("Multi-Head Attention Optimizer failed to parse Huggingface BERT attention Mask Pattern.");
            return TNN_OK;
        }

        // 1.2: Attention Mask Ptterns
        std::vector<std::shared_ptr<Graph>> mha_patterns;

        std::string module_linear_str = R"(
            graph(%in):
                %qkv_linear_mm      = MatMul(%in)
                %qkv_linear_add     = Add(%qkv_linear_mm)
            return (%qkv_linear_add)
        )";
        std::shared_ptr<Graph> module_linear_pattern = nullptr;
        if (graph_parser.parseFromString(module_linear_str)) {
            module_linear_pattern = graph_parser.getGraph();
            if (!registry.registerGraph("Module_Linear", module_linear_pattern)) {
                WARN("Fail to Register SubModule: Linear into Graph Registry.");
                return TNN_OK;
            }
        } else {
            WARN("Multi-Head Attention Optimizer failed to parse from MHA huggingface pattern string.");
            return TNN_OK;
        }

        std::string module_size_one_dim_str = R"(
            graph(%target_tensor):
                %tensor_shape        = Shape(%target_tensor)
                %shape_gather        = Gather(%tensor_shape)
                %shape_unsqueeze     = Unsqueeze(%shape_gather)
            return (%shape_unsqueeze)
        )";
        std::shared_ptr<Graph> module_size_one_dim_pattern = nullptr;
        if (graph_parser.parseFromString(module_size_one_dim_str)) {
            module_size_one_dim_pattern = graph_parser.getGraph();
            if (!registry.registerGraph("Module_Size_One_Dim", module_size_one_dim_pattern)) {
                WARN("Fail to Register SubModule: Size[dim] into Graph Registry.");
                return TNN_OK;
            }
        } else {
            WARN("Multi-Head Attention Optimizer failed to parse from MHA huggingface pattern string.");
            return TNN_OK;
        }

        std::string pattern_huggingface_bert_mha_str = R"(
            graph(%in, %attn_mask, %num_heads, %per_head_size, %hidden_size):
                %q_linear_add         = Module_Linear(%in)
                %k_linear_add         = Module_Linear(%in)
                %v_linear_add         = Module_Linear(%in)
                %q_batch              = Module_Size_One_Dim(%q_linear_add)
                %q_seqlen             = Module_Size_One_Dim(%q_linear_add)
                %q_reshape_shape      = Concat(%q_batch, %q_seqlen, %num_heads, %per_head_size)
                %q_reshape            = ReshapeTorch(%q_linear_add, %q_reshape_shape)
                %q_permute            = Permute(%q_reshape)
                %k_batch              = Module_Size_One_Dim(%k_linear_add)
                %k_seqlen             = Module_Size_One_Dim(%k_linear_add)
                %k_reshape_shape      = Concat(%k_batch, %k_seqlen, %num_heads, %per_head_size)
                %k_reshape            = ReshapeTorch(%k_linear_add, %k_reshape_shape)
                %k_permute            = Permute(%k_reshape)
                %v_batch              = Module_Size_One_Dim(%v_linear_add)
                %v_seqlen             = Module_Size_One_Dim(%v_linear_add)
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
                %ac_batch             = Module_Size_One_Dim(%attn_context_permute)
                %ac_seqlen            = Module_Size_One_Dim(%attn_context_permute)
                %ac_reshape_shape     = Concat(%ac_batch, %ac_seqlen, %hidden_size)
                %attn_context_reshape = ReshapeTorch(%attn_context_permute, %ac_reshape_shape)
                return (%attn_context_reshape)
        )";
        if (graph_parser.parseFromString(pattern_huggingface_bert_mha_str)) {
            mha_patterns.push_back(graph_parser.getGraph());
        } else {
            WARN("Multi-Head Attention Optimizer failed to parse Huggingface BERT Multi-Head Attention Pattern.");
            return TNN_OK;
        }
        // FOR DEBUGGING
        //TNN_NS::Logger::instance().set_verbose_level("I");



        // [2]. Global Pattern Match Check
        //      Graph will not be rewrited in this step.
        // 2.1: Attention Mask
        std::shared_ptr<AnchorGraph> matched_attn_mask_pattern = GetMatchedMHAAttentionMask(graph, attn_mask_patterns, resource);
        if (!matched_attn_mask_pattern) {
            return TNN_OK;
        }

        // 2.2: Multi-Head Attention Main Part.
        int num_heads     = -1;
        int per_head_size = -1;
        int hidden_size   = -1;
        std::vector<std::shared_ptr<AnchorGraph>> matched_mha_patterns = GetMatchedMHAMainParts(graph, mha_patterns, resource, num_heads, per_head_size, hidden_size);
        if (matched_mha_patterns.empty()) {
            return TNN_OK;
        }

        // 2.3: Combination of Attention Mask and Multi-Head Attention
        bool attn_mask_mha_connected = CheckAttnMaskMainPartConnection(matched_attn_mask_pattern, matched_mha_patterns);
        if (!attn_mask_mha_connected) {
            return TNN_OK;
        }



        // [3]. Rewrite Graph,
        //      In this step, graph would be rewrited.
        // 3.1: Rewrite Attention Mask
        //      for dense QKV to Context TRT Plugin.
        bool attn_mask_updated = rewriteAttentionMask(graph, matched_attn_mask_pattern);
        if (!attn_mask_updated) {
            return TNN_OK;
        }

        // 3.2: Rewrite Multi-Head Attention
        //      for dense QKV to Context TRT Plugin.
        bool mha_main_body_updated = rewriteMultiHeadAttention(graph, mha_patterns, resource, num_heads, per_head_size, hidden_size);
        if (!mha_main_body_updated) {
            return TNN_OK;
        }

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
