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
#include "tnn/utils/omp_utils.h"
#include <xbyak/xbyak.h>

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
        dim_t cur_m = MIN(M - i, conv_gemm_conf.kernel_m_r_);

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
                conv_gemm_conf.kernels_[16][N](K, cur_a, lda, cur_b, N / 6, cur_c, ldc, bias, first, act_type);
                i+=16;
                break;
        }
    }
}

// sgemm col_major a no_trans, b no_trans
// src_a: M * K, lda = M
// src_b: K * N, ldb = K
// dst  : M * N, ldc = M
void conv_sgemm_nn_col_major(
        dim_t M, dim_t N, dim_t K,
        const float * src_a, dim_t lda,
        const float * src_b, dim_t ldb,
        float * dst, dim_t ldc,
        const float * bias, dim_t act_type,
        float *pack_buf,
        conv_gemm_config<float, float, float> &conv_gemm_conf)
{
    dim_t M_c = conv_gemm_conf.M_c_;
    dim_t K_c = conv_gemm_conf.K_c_;
    dim_t m_block = conv_gemm_conf.m_block_;
    dim_t n_block = conv_gemm_conf.n_block_;

    dim_t i, j, k;
    i = j = k = 0;

    dim_t first = 0;
    dim_t post_type;

    auto pack_a_buf = pack_buf;
    auto pack_b_buf = pack_buf + divUp(M_c * K_c * sizeof(float), 32) / sizeof(float);

    // if no bias, first set to 1, load c from dst
    if (bias == nullptr) {
        first = 1;
    }

    for (k = 0; k < K; k += K_c)  {
        if (k + K_c >= K) {
            post_type = act_type;
        } else {
            post_type = 0;
        }

        dim_t cur_k = MIN(K - k, K_c);

        // pack b -> K_c * N;
        // const float *pack_b_k = src_b + k * divUp(N, n_block);
        pack_col_b_n(src_b + k, ldb, pack_b_buf, K_c, cur_k, N, conv_gemm_conf);

        for (i = 0; i < M; i += M_c)  {
            dim_t cur_m = MIN(M - i, M_c);
            // pack a -> M_c * K_c;
            pack_col_a_n(src_a + i + k * lda, lda, pack_a_buf, K_c, cur_k, cur_m, conv_gemm_conf);

            for (j = 0; j < N;)  {
                dim_t cur_n = MIN(N - j, conv_gemm_conf.kernel_n_r_);
                float * cur_c = dst + i + j * ldc;

                const float * packed_cur_b = pack_b_buf + divDown(j, n_block) * K_c + j % n_block;
                const float * cur_bias = bias + j;
                conv_sgemm_block_n(cur_m, cur_n, cur_k, pack_a_buf, lda, packed_cur_b, ldb, cur_c, ldc, cur_bias, first, post_type, conv_gemm_conf);
                j += cur_n;
            }
        }
        // if k != 0, first = 1
        first = 1;
    }
}

// sgemm col_major a trans, b no_trans
// src_a: K * M, lda = K
// src_b: K * N, ldb = K, prepacked
// dst  : M * N, ldc = M
void conv_sgemm_tn_col_major_prepack_b(
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

    dim_t first = 0;
    dim_t post_type;

    // if no bias, first set to 1, load c from dst
    if (bias == nullptr) {
        first = 1;
    }

    for (k = 0; k < K; k += K_c)  {
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
            pack_col_a_t(src_a + k + i * lda, lda, src_trans_buf, K_c, cur_k, cur_m, conv_gemm_conf);

            for (j = 0; j < N;)  {
                dim_t cur_n = MIN(N - j, conv_gemm_conf.kernel_n_r_);
                float * cur_c = dst + i + j * ldc;

                const float * packed_cur_b = pack_b_k + divDown(j, n_block) * K_c + j % n_block;
                const float * cur_bias = bias + j;
                conv_sgemm_block_n(cur_m, cur_n, cur_k, src_trans_buf, lda, packed_cur_b, ldb, cur_c, ldc, cur_bias, first, post_type, conv_gemm_conf);
                j += cur_n;
            }
        }
        // if k != 0, first = 1
        first = 1;
    }
}

// sgemm col_major a trans, b no_trans
// src_a: K * M, lda = K, prepacked
// src_b: K * N, ldb = K
// dst  : M * N, ldc = M
void conv_sgemm_tn_col_major_prepack_a(
        dim_t M, dim_t N, dim_t K,
        const float * src_a, dim_t lda,
        const float * src_b, dim_t ldb,
        float * dst, dim_t ldc,
        const float * bias, dim_t act_type,
        float *pack_b_buf,
        conv_gemm_config<float, float, float> &conv_gemm_conf)
{
    dim_t M_c = conv_gemm_conf.M_c_;
    dim_t K_c = conv_gemm_conf.K_c_;
    dim_t m_block = conv_gemm_conf.m_block_;
    dim_t n_block = conv_gemm_conf.n_block_;

    dim_t first = 0;
    dim_t post_type;

    // if no bias, first set to 1, load c from dst
    if (bias == nullptr) {
        first = 1;
    }

    for (dim_t k = 0; k < K; k += K_c)  {
        if (k + K_c >= K) {
            post_type = act_type;
        } else {
            post_type = 0;
        }

        dim_t cur_k = MIN(K - k, K_c);

        // pack b -> K_c * N;
        pack_col_b_n(src_b + k, ldb, pack_b_buf, K_c, cur_k, N, conv_gemm_conf);

        OMP_PARALLEL_FOR_DYNAMIC_
        for (dim_t i = 0; i < M; i += M_c)  {
            dim_t cur_m = MIN(M - i, M_c);
            // pack a -> M_c * K_c;
            auto src_a_i = src_a + k * divUp(M, m_block) + i * K_c;

            for (dim_t j = 0; j < N;)  {
                dim_t cur_n = MIN(N - j, conv_gemm_conf.kernel_n_r_);
                float * cur_c = dst + i + j * ldc;

                const float * packed_cur_b = pack_b_buf + divDown(j, n_block) * K_c + j % n_block;
                const float * cur_bias = bias + j;
                conv_sgemm_block_n(cur_m, cur_n, cur_k, src_a_i, lda, packed_cur_b, ldb, cur_c, ldc, cur_bias, first, post_type, conv_gemm_conf);
                j += cur_n;
            }
        }
        // if k != 0, first = 1
        first = 1;
    }
}

// pack col major B no_trans [K x N]
void conv_pack_col_b_n(
        dim_t N, dim_t K,
        const float * src, dim_t ld_src,
        float * dst,
        conv_gemm_config<float, float, float> &conv_gemm_conf)
{
    // dim_t K_c = conv_gemm_conf.K_c_;
    dim_t K_c = K;
    dim_t n_block = conv_gemm_conf.n_block_;
    dim_t N_round_up = divUp(N, n_block);

    for (dim_t k = 0; k < K; k += K_c) {
        dim_t cur_k = MIN(K - k, K_c);
        float *pack_b = dst + k * N_round_up;
        pack_col_b_n(src + k, ld_src, pack_b, K_c, cur_k, N, conv_gemm_conf);
    }
}

// pack col major A trans [K x M]
void conv_pack_col_a_t(
    dim_t M, dim_t K,
    const float * src, dim_t lda,
    float * dst,
    conv_gemm_config<float, float, float> &conv_gemm_conf)
{
    dim_t M_c = conv_gemm_conf.M_c_;
    // dim_t K_c = conv_gemm_conf.K_c_;
    dim_t K_c = K;
    dim_t m_block = conv_gemm_conf.m_block_;
    dim_t n_block = conv_gemm_conf.n_block_;

    dim_t i, j, k;
    i = j = k = 0;

    for (k = 0; k < K; k += K_c)  {
        dim_t cur_k = MIN(K - k, K_c);
        auto src_k = src + k;
        auto dst_k = dst + k * divUp(M, m_block);

        for (i = 0; i < M; i += M_c)  {
            dim_t cur_m = MIN(M - i, M_c);
            // pack a -> M_c * K_c;
            pack_col_a_t(src_k + i * lda, lda, dst_k + i * K_c, K_c, cur_k, cur_m, conv_gemm_conf);
        }
    }
}

// // pack A [K * M]
// void conv_pack_a_n()
// {

// }

void conv_ajust_m_blk_size(
    int max_num_threads,
    dim_t m_all,
    dim_t &m_blk)
{
    // for 32bit, min M blk = 8
    // for 64bit, min M blk = 16
    int m_min = 8;
#ifdef XBYAK64
    m_min = 16;
#endif

    while ((m_all / m_blk) < max_num_threads &&
           m_blk > m_min) {
        m_blk = MAX(m_blk / 2, m_min);
    }
}

#define SGEMM_TILE_N 6

void set_block_size(int l2_size, const int N, const int M, const int K, int byte_size,
                    conv_gemm_config<float, float, float> &conv_gemm_conf) {
    int tile_n = conv_gemm_conf.n_block_;
    int tile_m = conv_gemm_conf.m_block_;
    const int l1cache = 32 * 1024 / byte_size;
    if (N >= M) {
        // inner kernel also a first, safe in l1 cache
        conv_gemm_conf.M_c_ = MAX(l1cache / K - tile_n, 1);
        // b safe in l2 cache
        int l2_size_b = l2_size / K - conv_gemm_conf.M_c_;
        conv_gemm_conf.N_c_ = MIN(l2_size_b, M);
    } else {
        if (N < l2_size / K - tile_n) {
            conv_gemm_conf.M_c_ = N;
        } else {
            conv_gemm_conf.N_c_ = MAX(l2_size / K - tile_n, 1);
        }
        conv_gemm_conf.N_c_ = tile_n;
    }
    conv_gemm_conf.N_c_ = ROUND_UP(conv_gemm_conf.N_c_, tile_n);
    conv_gemm_conf.M_c_ = ROUND_UP(conv_gemm_conf.M_c_, tile_m);
}

template <int sgemm_tile_m>
void load_repack_a_impl(float *dst, const float *src, int width, int src_z_step, int ic) {
    int loop   = width / sgemm_tile_m;
    int remain = width % sgemm_tile_m;

    for (int db = 0; db < loop; db++) {
        auto src_b = src + db * sgemm_tile_m;
        auto dst_b = dst + db * sgemm_tile_m * ic;
        for (int c_i = 0; c_i < ic; c_i++) {
#ifdef __AVX2__
            if (sgemm_tile_m == 16) {
                _mm256_storeu_ps(dst_b, _mm256_loadu_ps(src_b));
                _mm256_storeu_ps(dst_b + 8, _mm256_loadu_ps(src_b + 8));
            } else if (sgemm_tile_m == 8) {
                _mm256_storeu_ps(dst_b, _mm256_loadu_ps(src_b));
            } else if (sgemm_tile_m == 4) {
                _mm_storeu_ps(dst_b, _mm_loadu_ps(src_b));
            }
#else
            memcpy(dst_b, src_b, sgemm_tile_m * sizeof(float));
#endif
            src_b += src_z_step;
            dst_b += sgemm_tile_m;
        }
    }
    {
        if (remain > 0) {
            auto src_b = src + loop * sgemm_tile_m;
            auto dst_b = dst + loop * sgemm_tile_m * ic;
            for (int c_i = 0; c_i < ic; c_i++) {
                memcpy(dst_b, src_b, remain * sizeof(float));
                src_b += src_z_step;
                dst_b += sgemm_tile_m;
            }
        }
    }
}

void load_repack_a(float *dst, const float *src, int width, int src_z_step, int ic, int tile_m) {
    if (tile_m == 16) {
        load_repack_a_impl<16>(dst, src, width, src_z_step, ic);
    } else if (tile_m == 8) {
        load_repack_a_impl<8>(dst, src, width, src_z_step, ic);
    } else if (tile_m == 4) {
        load_repack_a_impl<4>(dst, src, width, src_z_step, ic);
    }
}

template <int tile_m>
void gemm_unit_kernel_(dim_t K, const float *src_a, dim_t lda, const float *src_b,
                       dim_t n_loop, float *dst, dim_t ldc, const float *bias,
                       tnn::dim_t first, tnn::dim_t act_type,
                       conv_gemm_config<float, float, float> &conv_gemm_conf) {
    auto gemm_unit_impl_kernel = conv_gemm_conf.kernels_[tile_m][SGEMM_TILE_N];
    for (int n_i = 0; n_i < n_loop; n_i++) {
        gemm_unit_impl_kernel(K, src_a, lda, src_b, n_loop, dst, ldc, bias, first, act_type);
        dst += SGEMM_TILE_N * ldc;
        src_b += SGEMM_TILE_N * K;
        if (bias != nullptr) {
            bias += SGEMM_TILE_N;
        }
    }
}

template <>
void gemm_unit_kernel_<16>(dim_t K, const float *src_a, dim_t lda, const float *src_b,
                       dim_t n_loop, float *dst, dim_t ldc, const float *bias,
                       tnn::dim_t first, tnn::dim_t act_type,
                       conv_gemm_config<float, float, float> &conv_gemm_conf) {
    conv_gemm_conf.kernels_[16][SGEMM_TILE_N](
        K, src_a, lda, src_b, n_loop, dst, ldc, bias, first, act_type);
}

// weight[n, k] x src[m, k]
void gemm_compute_unit(float *dst, const float *src, const float *weight,
                       int k, int m, int n_blk, int x_loop, const float *bias, int act_type,
                       conv_gemm_config<float, float, float> &conv_gemm_conf) {
    int n_blk_unit = n_blk / SGEMM_TILE_N * SGEMM_TILE_N;
    int n_blk_left = n_blk - n_blk_unit;
    int tile_m     = conv_gemm_conf.m_block_;

    auto gemm_unit_kernel = gemm_unit_kernel_<16>;
    if (tile_m == 8) {
        gemm_unit_kernel = gemm_unit_kernel_<8>;
    } else if (tile_m == 4) {
        gemm_unit_kernel = gemm_unit_kernel_<4>;
    }
    auto gemm_left_kernel = conv_gemm_conf.kernels_[tile_m][n_blk_left];

    for (int x_i = 0; x_i < x_loop; x_i++) {
        auto dst_x = dst + x_i * tile_m;
        auto src_x = src + x_i * tile_m * k;

        gemm_unit_kernel(k, src_x, m, weight, n_blk_unit / SGEMM_TILE_N, dst_x, m, bias, 0, act_type, conv_gemm_conf);
        if (n_blk_left > 0) {
            auto dst_n = dst_x + n_blk_unit * m;
            auto weight_n = weight + n_blk_unit * k;
            auto bias_n = bias ? bias + n_blk_unit : nullptr;
            gemm_left_kernel(k, src_x, m, weight_n, k, dst_n, m, bias_n, 0, act_type);
        }
    }
}

void gemm_compute_left(float *dst, const float *src, const float *weight,
                       int k, int m, int n_blk, int m_blk, const float *bias, int act_type,
                       conv_gemm_config<float, float, float> &conv_gemm_conf) {
    int n_blk_unit = n_blk / SGEMM_TILE_N * SGEMM_TILE_N;
    int n_blk_left = n_blk - n_blk_unit;

    auto gemm_m1_unit_kernel = conv_gemm_conf.kernels_[1][SGEMM_TILE_N];
    auto gemm_m2_unit_kernel = conv_gemm_conf.kernels_[2][SGEMM_TILE_N];
    auto gemm_m4_unit_kernel = conv_gemm_conf.kernels_[4][SGEMM_TILE_N];
    auto gemm_m8_unit_kernel = conv_gemm_conf.kernels_[8][SGEMM_TILE_N];
    auto gemm_m1_left_kernel = conv_gemm_conf.kernels_[1][n_blk_left];
    auto gemm_m2_left_kernel = conv_gemm_conf.kernels_[2][n_blk_left];
    auto gemm_m4_left_kernel = conv_gemm_conf.kernels_[4][n_blk_left];
    auto gemm_m8_left_kernel = conv_gemm_conf.kernels_[8][n_blk_left];

    for (int n_i = 0; n_i < n_blk_unit; n_i += SGEMM_TILE_N) {
        auto dst_n = dst + n_i * m;
        auto weight_n = weight + n_i * k;
        auto bias_n = bias ? bias + n_i : nullptr;
        for (int x_i = 0; x_i < m_blk;) {
            int cur_x = MIN(m_blk - x_i, conv_gemm_conf.kernel_m_r_);
            auto dst_x = dst_n + x_i;
            auto src_x = src + x_i;
            if (cur_x >= 8) {
                gemm_m8_unit_kernel(k, src_x, m, weight_n, k, dst_x, m, bias_n, 0, act_type);
                x_i += 8;
            } else if (cur_x >= 4) {
                gemm_m4_unit_kernel(k, src_x, m, weight_n, k, dst_x, m, bias_n, 0, act_type);
                x_i += 4;
            } else if (cur_x >= 2) {
                gemm_m2_unit_kernel(k, src_x, m, weight_n, k, dst_x, m, bias_n, 0, act_type);
                x_i += 2;
            } else {
                gemm_m1_unit_kernel(k, src_x, m, weight_n, k, dst_x, m, bias_n, 0, act_type);
                x_i += 1;
            }
        }
    }
    if (n_blk_left > 0) {
        auto dst_n = dst + n_blk_unit * m;
        auto weight_n = weight + n_blk_unit * k;
        auto bias_n = bias ? bias + n_blk_unit : nullptr;
        for (int x_i = 0; x_i < m_blk;) {
            int cur_x = MIN(m_blk - x_i, conv_gemm_conf.kernel_m_r_);
            auto dst_x = dst_n + x_i;
            auto src_x = src + x_i;
            if (cur_x >= 8) {
                gemm_m8_left_kernel(k, src_x, m, weight_n, k, dst_x, m, bias_n, 0, act_type);
                x_i += 8;
            } else if (cur_x >= 4) {
                gemm_m4_left_kernel(k, src_x, m, weight_n, k, dst_x, m, bias_n, 0, act_type);
                x_i += 4;
            } else if (cur_x >= 2) {
                gemm_m2_left_kernel(k, src_x, m, weight_n, k, dst_x, m, bias_n, 0, act_type);
                x_i += 2;
            } else {
                gemm_m1_left_kernel(k, src_x, m, weight_n, k, dst_x, m, bias_n, 0, act_type);
                x_i += 1;
            }
        }
    }
}

// sgemm col_major a no_trans, b no_trans
// src_a: M * K, lda = M
// src_b: K * N, ldb = K, prepacked
// dst  : M * N, ldc = M
// lhs  : M > N
void conv_sgemm_nn_col_major_prepack_b_lhs(
        dim_t M, dim_t N, dim_t K,
        const float * src_a, dim_t lda,
        const float * src_b, dim_t ldb,
        float * dst, dim_t ldc,
        const float * bias, dim_t act_type,
        float *src_trans_buf,
        conv_gemm_config<float, float, float> &conv_gemm_conf)
{
    dim_t M_c = conv_gemm_conf.M_c_;
    dim_t N_c = conv_gemm_conf.N_c_;
    dim_t m_block = conv_gemm_conf.m_block_;
    dim_t n_block = conv_gemm_conf.n_block_;

    dim_t m_loop   = M / M_c;
    dim_t m_remain = M % M_c;

    OMP_PARALLEL_FOR_
    for (dim_t m_i = 0; m_i <= m_loop; m_i++) {
        int thread_id = OMP_TID_;
        auto src_a_repack = src_trans_buf + thread_id * M_c * K;
        auto src_a_m = src_a + m_i * M_c;
        auto m_eff = (m_i < m_loop) ? M_c : m_remain;
        auto x_loop = m_eff / m_block;
        auto x_remain = m_eff % m_block;

        load_repack_a(src_a_repack, src_a_m, m_eff, M, K, m_block);

        for (dim_t n_i = 0; n_i < N; n_i += N_c) {
            auto n_eff = MIN(N_c, N - n_i);
            auto src_b_n = src_b + n_i * K;
            auto dst_n = dst + n_i * M + m_i * M_c;
            auto bias_n = bias ? bias + n_i : nullptr;
            {
                gemm_compute_unit(dst_n,
                                  src_a_repack,
                                  src_b_n, K, M, n_eff,
                                  x_loop, bias_n, act_type, conv_gemm_conf);
            }
            if (x_remain > 0) {
                gemm_compute_left(dst_n + x_loop * m_block,
                                  src_a_repack + x_loop * m_block * K,
                                  src_b_n, K, M, n_eff,
                                  x_remain, bias_n, act_type, conv_gemm_conf);
            }
        }
    }
}

// sgemm col_major a no_trans, b no_trans
// src_a: M * K, lda = M
// src_b: K * N, ldb = K, prepacked
// dst  : M * N, ldc = M
// rhs  : M <= N
void conv_sgemm_nn_col_major_prepack_b_rhs(
        dim_t M, dim_t N, dim_t K,
        const float * src_a, dim_t lda,
        const float * src_b, dim_t ldb,
        float * dst, dim_t ldc,
        const float * bias, dim_t act_type,
        float *src_trans_buf,
        conv_gemm_config<float, float, float> &conv_gemm_conf)
{
    dim_t M_c = conv_gemm_conf.M_c_;
    dim_t N_c = conv_gemm_conf.N_c_;
    dim_t m_block = conv_gemm_conf.m_block_;
    dim_t n_block = conv_gemm_conf.n_block_;

    dim_t m_loop   = M / M_c;
    dim_t m_remain = M % M_c;

    for (dim_t m_i = 0; m_i <= m_loop; m_i++) {
        auto src_a_repack = src_trans_buf;
        auto src_a_m = src_a + m_i * M_c;
        auto m_eff = (m_i < m_loop) ? M_c : m_remain;
        auto x_loop = m_eff / m_block;
        auto x_remain = m_eff % m_block;

        load_repack_a(src_a_repack, src_a_m, m_eff, M, K, m_block);

        OMP_PARALLEL_FOR_
        for (dim_t n_i = 0; n_i < N; n_i += N_c) {
            auto n_eff = MIN(N_c, N - n_i);
            auto src_b_n = src_b + n_i * K;
            auto dst_n = dst + n_i * M + m_i * M_c;
            auto bias_n = bias ? bias + n_i : nullptr;
            {
                gemm_compute_unit(dst_n,
                                  src_a_repack,
                                  src_b_n, K, M, n_eff,
                                  x_loop, bias_n, act_type, conv_gemm_conf);
            }
            if (x_remain > 0) {
                gemm_compute_left(dst_n + x_loop * m_block,
                                  src_a_repack + x_loop * m_block * K,
                                  src_b_n, K, M, n_eff,
                                  x_remain, bias_n, act_type, conv_gemm_conf);
            }
        }
    }
}

// sgemm col_major a no_trans, b no_trans
// src_a: M * K, lda = M
// src_b: K * N, ldb = K, prepacked
// dst  : M * N, ldc = M
void conv_sgemm_nn_col_major_prepack_b(
        dim_t M, dim_t N, dim_t K,
        const float * src_a, dim_t lda,
        const float * src_b, dim_t ldb,
        float * dst, dim_t ldc,
        const float * bias, dim_t act_type,
        float *src_trans_buf,
        conv_gemm_config<float, float, float> &conv_gemm_conf)
{
    if (M > N) {
        conv_sgemm_nn_col_major_prepack_b_lhs(M, N, K,
            src_a, lda, src_b, ldb, dst, ldc, bias, act_type,
            src_trans_buf, conv_gemm_conf);
    } else {
        conv_sgemm_nn_col_major_prepack_b_rhs(M, N, K,
            src_a, lda, src_b, ldb, dst, ldc, bias, act_type,
            src_trans_buf, conv_gemm_conf);
    }
}

} // namespace tnn
