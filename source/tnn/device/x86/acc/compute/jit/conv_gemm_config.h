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

#ifndef TNN_JIT_CONV_GEMM_CONFIG_HPP_
#define TNN_JIT_CONV_GEMM_CONFIG_HPP_

#include "tnn/device/x86/acc/compute/jit/common/type_def.h"

namespace TNN_NS {

template<typename a_t, typename b_t, typename c_t>
struct conv_gemm_config {
    
typedef void (*fetch_n_func_t)(const dim_t m, const a_t * a, const dim_t lda, a_t * b, const dim_t ldb, const dim_t block_size);
typedef void (*fetch_t_func_t)(const dim_t m, const a_t * a, const dim_t lda, a_t * b, const dim_t ldb, const dim_t block_size);
typedef void (*conv_sgemm_ker_func_t)(const dim_t K,
                                 const a_t * src_a, dim_t lda,
                                 const b_t * src_b, dim_t ldb,
                                 c_t * dst, dim_t ldc,
                                 const b_t * bias, dim_t first, dim_t act_type);

    conv_gemm_config(const dim_t m_block = 16, const dim_t n_block = 6);

    // block size for data packing
    dim_t m_block_;
    dim_t n_block_;

    // block size for kernel register blocking
    dim_t kernel_m_r_;
    dim_t kernel_n_r_;

    // block size for matrix spliting
    dim_t M_c_;
    dim_t K_c_;

    constexpr static int nb_kernels_m = 16;
    constexpr static int nb_kernels_n = 6;

    fetch_t_func_t pack_t_ker_[nb_kernels_m + 1];
    fetch_t_func_t pack_t_4x16_ker_;
    fetch_t_func_t pack_t_4x8_ker_;

    fetch_n_func_t pack_n_ker_[nb_kernels_n + 1];

    conv_sgemm_ker_func_t kernels_[nb_kernels_m + 1][nb_kernels_n + 1];

private:
    void init_jit_kernel();
};

} // namespace tnn

#endif // TNN_JIT_GEMM_CONFIG_HPP_