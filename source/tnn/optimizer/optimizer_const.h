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

}

#endif // TNN_SOURCE_TNN_OPTIMIZER_OPTIMIZER_CONST_H_
