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

#ifdef __arm__
#ifndef __aarch64__

#include "tnn/device/arm/acc/compute/asm_func_name.S"

.text
.align 5

asm_function GEMM_FP16_N8
//void GEMM_FP16_N8(float* dst, const float* src, const float* weight, int src_depth,
//                            int dst_step, int dst_depth, int width, float *bias, int64_t relu)
//Auto Load:
//x0:dst, x1:src, x2:weight, x3:src_depth, x4:dst_step, x5:dst_depth, x6: width, x7: bias

#endif
#endif
