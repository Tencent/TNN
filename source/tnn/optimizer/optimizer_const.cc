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

#include "tnn/optimizer/optimizer_const.h"

#include "tnn/core/macro.h"

namespace TNN_NS {

const char * kNetOptimizerFuseConvPost =
    "net_optimizer_fuse_conv_post";

const char * kNetOptimizerFuseConvActivation =
    "net_optimizer_fuse_conv_activation";

const char * kNetOptimizerFuseConvAdd =
    "net_optimizer_fuse_conv_add";

const char * kNetOptimizerCbamFusedReduce =
    "net_optimizer_cbam_fused_reduce";

const char * kNetOptimizerCbamFusedPooling =
    "net_optimizer_cbam_fused_pooling";

const char * kNetOptimizerInsertInt8Reformat =
    "net_optimizer_insert_int8_reformat";

const char * kNetOptimizerInsertFp16Reformat =
    "net_optimizer_insert_fp16_reformat";

const char * kNetOptimizerInsertLayoutReformat =
    "net_optimizer_insert_layout_reformat";

const char * kNetOptimizerRemoveLayers =
    "net_optimizer_remove_layers";

const char * kNetOptimizerConvertInt8Layers =
    "net_optimizer_convert_int8_layers";

const char * kNetOptimizerDynamicRangeDequant =
    "net_optimizer_dynamic_range_dequant";

const char * kNetOptimizerConvertMatMulToConv =
    "net_optimizer_convert_matmul_to_conv";

}  // namespace TNN_NS
