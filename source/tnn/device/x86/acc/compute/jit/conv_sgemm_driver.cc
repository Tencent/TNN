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

#include "tnn/device/x86/acc/compute/jit/cblas.h"

#include <stdio.h>

#include "tnn/device/x86/acc/compute/jit/common/type_def.h"
#include "tnn/device/x86/acc/compute/jit/utils/utils.h"
#include "tnn/device/x86/acc/compute/jit/conv_data_packing.h"
#include "tnn/device/x86/acc/compute/jit/conv_gemm_config.h"
#include "tnn/device/x86/acc/compute/jit/utils/timer.hpp"
#include "tnn/device/x86/acc/compute/jit/conv_sgemm_driver.h"

namespace TNN_NS {

void conv_sgemm_block_n(
        dim_t M, dim_t N, dim_t K,
        const float * src_a, dim_t lda,
        const float * src_b, dim_t ldb,
        float * dst, dim_t ldc,
        const float * bias, dim_t first, dim_t act_type,
        conv_gemm_config<float, float, float> &conv_gemm_conf) 
{

    dim_t K_c = conv_gemm_conf.K_c_;
    dim_t m_block = conv_gemm_conf.m_block_;

    for(dim_t i=0;i<M;)  {
        dim_t cur_m = std::min(M - i, conv_gemm_conf.kernel_m_r_);

        const float * cur_a = src_a + divDown(i, m_block) * K_c + i % m_block;
        const float * cur_b = src_b;
        float * cur_c = dst + i;

        switch(cur_m) {
            case 1:
                conv_gemm_conf.kernels_[1][N](K, cur_a, lda, cur_b, ldb, cur_c, ldc, bias, first, act_type);
                i+=1;
                break;
            case 2:
            case 3:
                conv_gemm_conf.kernels_[2][N](K, cur_a, lda, cur_b, ldb, cur_c, ldc, bias, first, act_type);
                i+=2;
                break;
            case 4:
            case 5:
            case 6:
            case 7:
                conv_gemm_conf.kernels_[4][N](K, cur_a, lda, cur_b, ldb, cur_c, ldc, bias, first, act_type);
                i+=4;
                break;
            case 8:
            case 9:
            case 10:
            case 11:
            case 12:
            case 13:
            case 14:
            case 15:
                conv_gemm_conf.kernels_[8][N](K, cur_a, lda, cur_b, ldb, cur_c, ldc, bias, first, act_type);
                i+=8;
                break;
            default:
                conv_gemm_conf.kernels_[16][N](K, cur_a, lda, cur_b, ldb, cur_c, ldc, bias, first, act_type);
                i+=16;
                break;
        }
    }
}

void conv_sgemm_nn_col_major(
        dim_t M, dim_t N, dim_t K,
        const float * src_a, dim_t lda,
        const float * src_b, dim_t ldb,
        float * dst, dim_t ldc,
        const float * bias, dim_t act_type,
        float *src_trans_buf,
        conv_gemm_config<float, float, float> &conv_gemm_conf)
{
    dim_t M_c = conv_gemm_conf.M_c_;
    dim_t K_c = conv_gemm_conf.K_c_;
    dim_t m_block = conv_gemm_conf.m_block_;
    dim_t n_block = conv_gemm_conf.n_block_;

    dim_t i, j, k;
    i = j = k = 0;

    dim_t first;
    dim_t post_type;

    for (k = 0; k < K; k += K_c)  {
        if (k == 0) {
            first = 0;
        } else {
            first = 1;
        }

        if (k + K_c >= K) {
            post_type = act_type;
        } else {
            post_type = 0;
        }

        dim_t cur_k = MIN(K - k, K_c);

        // pack b -> K_c * N;
        const float *pack_b_k = src_b + k * divUp(N, n_block);

        for (i = 0; i < M; i += M_c)  {
            dim_t cur_m = MIN(M - i, M_c);
            // pack a -> M_c * K_c;
            pack_t(src_a + i + k * lda, lda, src_trans_buf, K_c, cur_k, cur_m, conv_gemm_conf);

            for (j = 0; j < N;)  {
                dim_t cur_n = MIN(N - j, conv_gemm_conf.kernel_n_r_);
                float * cur_c = dst + i + j * ldc;

                const float * packed_cur_b = pack_b_k + divDown(j, n_block) * K_c + j % n_block;
                const float * cur_bias = bias + j;
                conv_sgemm_block_n(cur_m, cur_n, cur_k, src_trans_buf, lda, packed_cur_b, ldb, cur_c, ldc, cur_bias, first, post_type, conv_gemm_conf);
                j += cur_n;
            }
        }
    }
}

void conv_pack_weights(
        dim_t N, dim_t K,
        const float * src, dim_t ld_src,
        float * dst,
        conv_gemm_config<float, float, float> &conv_gemm_conf)
{
    dim_t K_c = conv_gemm_conf.K_c_;
    dim_t n_block = conv_gemm_conf.n_block_;
    dim_t N_round_up = divUp(N, n_block);

    for (dim_t k = 0; k < K; k += K_c) {
        dim_t cur_k = MIN(K - k, K_c);
        float *pack_b = dst + k * N_round_up;
        pack_n(src + k, ld_src, pack_b, K_c, cur_k, N, conv_gemm_conf);
    }
}

} // namespace tnn
