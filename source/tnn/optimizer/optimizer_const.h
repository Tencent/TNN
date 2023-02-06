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

#include "tnn/core/macro.h"

namespace TNN_NS {

extern const char * kNetOptimizerFuseConvPost;

extern const char * kNetOptimizerFuseConvActivation;

extern const char * kNetOptimizerFuseConvAdd;

extern const char * kNetOptimizerCbamFusedReduce;

extern const char * kNetOptimizerCbamFusedPooling;

extern const char * kNetOptimizerInsertInt8Reformat;

extern const char * kNetOptimizerInsertFp16Reformat;

extern const char * kNetOptimizerInsertLayoutReformat;

extern const char * kNetOptimizerRemoveLayers;

extern const char * kNetOptimizerConvertInt8Layers;

extern const char * kNetOptimizerDynamicRangeDequant;

extern const char * kNetOptimizerConvertMatMulToConv;

}

#endif // TNN_SOURCE_TNN_OPTIMIZER_OPTIMIZER_CONST_H_
