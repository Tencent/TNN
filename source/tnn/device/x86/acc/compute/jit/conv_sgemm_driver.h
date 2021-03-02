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

#ifndef SOURCE_TNN_DEVICE_X86_ACC_COMPUTE_JIT_CONV_SGEMM_DRIVER_H_
#define SOURCE_TNN_DEVICE_X86_ACC_COMPUTE_JIT_CONV_SGEMM_DRIVER_H_

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/device/x86/acc/compute/jit/conv_gemm_config.h"

namespace TNN_NS {

void conv_sgemm_nn_col_major(
        dim_t M, dim_t N, dim_t K,
        const float * src_a, dim_t lda,
        const float * src_b, dim_t ldb,
        float * dst, dim_t ldc,
        const float * bias, dim_t act_type,
        float * src_buf,
        conv_gemm_config<float, float, float> &conv_gemm_conf);

void conv_pack_weights(
        dim_t N, dim_t K,
        const float * src, dim_t ld_src,
        float * dst,
        conv_gemm_config<float, float, float> &conv_gemm_conf);

}   // namespace TNN_NS

#endif

