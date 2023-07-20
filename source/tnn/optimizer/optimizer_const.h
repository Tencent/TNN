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

#ifndef TNN_SOURCE_TNN_OPTIMIZER_OPTIMIZER_CONST_H_
#define TNN_SOURCE_TNN_OPTIMIZER_OPTIMIZER_CONST_H_

#include <string>

namespace TNN_NS {

static const std::string kNetOptimizerFuseConvPost =
    "net_optimizer_fuse_conv_post";

static const std::string kNetOptimizerFuseConvActivation =
    "net_optimizer_fuse_conv_activation";

static const std::string kNetOptimizerFuseConvAdd =
    "net_optimizer_fuse_conv_add";

static const std::string kNetOptimizerCbamFusedReduce =
    "net_optimizer_cbam_fused_reduce";

static const std::string kNetOptimizerCbamFusedPooling =
    "net_optimizer_cbam_fused_pooling";

static const std::string kNetOptimizerInsertInt8Reformat =
    "net_optimizer_insert_int8_reformat";

static const std::string kNetOptimizerInsertFp16Reformat =
    "net_optimizer_insert_fp16_reformat";

static const std::string kNetOptimizerInsertLayoutReformat =
    "net_optimizer_insert_layout_reformat";

static const std::string kNetOptimizerRemoveLayers =
    "net_optimizer_remove_layers";

static const std::string kNetOptimizerConvertInt8Layers =
    "net_optimizer_convert_int8_layers";

static const std::string kNetOptimizerDynamicRangeDequant =
    "net_optimizer_dynamic_range_dequant";

static const std::string kNetOptimizerQDQ = 
    "net_optimizer_qdq";

static const std::string kNetOptimizerContextMarker =
    "net_optimizer_context_marker";

static const std::string kNetOptimizerEffectiveTransformer =
    "net_optimizer_effective_transformer";

static const std::string kNetOptimizerFuseAddLayerNorm =
    "net_optimizer_fuse_add_layernorm";

static const std::string kNetOptimizerFuseFFN =
    "net_optimizer_fuse_ffn";

static const std::string kNetOptimizerFuseAttention =
    "net_optimizer_fuse_attention";

static const std::string kNetOptimizerFuseMatmulConcat =
    "net_optimizer_fuse_matmul_concat";

static const std::string kNetOptimizerQuantOptimizerGroup =
    "net_optimizer_quant_optimizer_group";

static const std::string kNetOptimizerFuseLayerNorm =
    "net_optimizer_fuse_layer_norm";

static const std::string kNetOptimizerRemoveInplaceOps =
    "net_optimizer_remove_inplace_ops";

static const std::string kNetOptimizerConvertMatMulToConv =
    "net_optimizer_convert_matmul_to_conv";

static const std::string kNetOptimizerFuseCrossAttention =
    "net_optimizer_fuse_cross_attention";

static const std::string kNetOptimizerFuseFlashAttention =
    "net_optimizer_fuse_flash_attention";

static const std::string kNetOptimizerFuseSplitGELU =
    "net_optimizer_fuse_split_gelu";

static const std::string kNetOptimizerFuseGroupNormSwish =
    "net_optimizer_fuse_group_norm_swish";

}

#endif // TNN_SOURCE_TNN_OPTIMIZER_OPTIMIZER_CONST_H_
