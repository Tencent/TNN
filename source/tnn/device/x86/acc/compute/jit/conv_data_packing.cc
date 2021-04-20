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


#include "tnn/device/x86/acc/compute/jit/data_packing.h"

#include <immintrin.h>
#include <xmmintrin.h>

#include <stdio.h>

#include "tnn/device/x86/acc/compute/jit/conv_gemm_config.h"

namespace TNN_NS {

//  pack block_size on non-leading dimension, n denotes no-transpose. 
//  eg. input:   A MxN matrix in col major, so the storage-format is (N, M) 
//      output:  B MxN matrix in col major(N-packed), so the storage-format is 
//                    (divUp(N, block_size), M, block_size)
template<typename T>
void pack_col_b_n(const T * a, dim_t lda, T * b, dim_t ldb, dim_t m, dim_t n, conv_gemm_config<T, T, T> &conv_gemm_conf)
{
    int block_size = conv_gemm_conf.n_block_;
    dim_t i=0; 
    for(;n-i>= block_size;i+= block_size) {
        const T * cur_a = a + i * lda;
        T * cur_b = b + i * ldb;
        conv_gemm_conf.pack_n_ker_[block_size](m, cur_a, lda, cur_b, ldb, block_size);
    }
    dim_t tail = n - i;
    if (tail > 0) {
        const T * cur_a = a + i * lda;
        T * cur_b = b + i * ldb;
        conv_gemm_conf.pack_n_ker_[tail](m, cur_a, lda, cur_b, ldb, block_size);
    }

}

template 
void pack_col_b_n<float>(const float * a, dim_t lda, float * b, dim_t ldb, 
                   dim_t m, dim_t n,
                   conv_gemm_config<float, float, float> &conv_gemm_conf);

//  pack block_size on leading dimension, t denotes transpose. 
//  eg. input:   A MxN matrix in row major, so the storage-format is (M, N) 
//      output:  B MxN matrix in col major(N-packed), so the storage-format is 
//                    (divUp(N, block_size), M, block_size)
template<typename T>
void pack_col_a_n(const T * a, dim_t lda, T * b, dim_t ldb, dim_t m, dim_t n, conv_gemm_config<T, T, T> &conv_gemm_conf) 
{
    dim_t block_size = conv_gemm_conf.m_block_;
    dim_t i = 0; 

    if (block_size == 16) {
        for(;i + 64 <=n;i+=64) {
            const T * cur_a = a + i;
            T * cur_b = b + i * ldb;
            conv_gemm_conf.pack_t_4x16_ker_(m, cur_a, lda, cur_b, ldb, block_size);
        }
    } else if (block_size == 8) {
        
    }

    for(;i<n;) {
        const T * cur_a = a + i;
        T * cur_b = b + divDown(i, block_size) * ldb + i % block_size;
        dim_t cur_n = std::min(n - i, block_size);
        switch(cur_n) {
            case 1:
                conv_gemm_conf.pack_t_ker_[1](m, cur_a, lda, cur_b, ldb, block_size);
                i+=1;
                break;
            case 2:
                conv_gemm_conf.pack_t_ker_[2](m, cur_a, lda, cur_b, ldb, block_size);
                i+=2;
                break;
            case 3:
                conv_gemm_conf.pack_t_ker_[3](m, cur_a, lda, cur_b, ldb, block_size);
                i+=3;
                break;
            case 4:
            case 5:
            case 6:
            case 7:
                conv_gemm_conf.pack_t_ker_[4](m, cur_a, lda, cur_b, ldb, block_size);
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
                conv_gemm_conf.pack_t_ker_[8](m, cur_a, lda, cur_b, ldb, block_size);
                i+=8;
                break;
            default:
                conv_gemm_conf.pack_t_ker_[16](m, cur_a, lda, cur_b, ldb, block_size);
                i+=16;
                break;
        }
    }
    /*
    if (block_n == 16) {

        int i=0;
        // must be a perfect loop, n % block_n = 0
        for(;i + 64 <=n;i+=64) {
            const float * cur_a = a + i;
            float * cur_b = b + i * ldb;
            sgemm_fetch_t_4x16(m, cur_a, lda, cur_b, ldb, 16);
        }

    }

    */
}

template 
void pack_col_a_n<float>(const float * a, dim_t lda, float * b, dim_t ldb, 
                   dim_t m, dim_t n,
                   conv_gemm_config<float, float, float> &conv_gemm_conf);

template <int blk, typename T>
void pack_a_t_trans(
    const T *src,
    dim_t lda,
    T *dst,
    dim_t cur_k,
    dim_t block_size)
{
    for (int j = 0; j < blk; j++) {
        auto src_j = src + j * lda;
        auto dst_j = dst + j;
        for (int k = 0; k < cur_k; k++) {
            dst_j[k * block_size] = src_j[k];
        }
    }
}

// lda -> total_k
// ldb -> m_block_size (M_c)
// cur_k
// cur_m
template<typename T>
void pack_col_a_t(
    const T *src_a,
    dim_t lda,
    T *src_b,
    dim_t ldb,
    dim_t cur_k,
    dim_t cur_m,
    conv_gemm_config<T, T, T> &conv_gemm_conf)
{
    dim_t block_size = conv_gemm_conf.m_block_;
    dim_t i = 0;

    if (block_size == 16) {
        for (; i + 15 < cur_m; i += 16) {
            auto a_ptr = src_a + i * lda;
            auto b_ptr = src_b + i * ldb;
            pack_a_t_trans<16, T>(a_ptr, lda, b_ptr, cur_k, block_size);
        }
    }
    for (; i + 7 < cur_m; i += 8) {
        auto a_ptr = src_a + i * lda;
        auto b_ptr = src_b + divDown(i, block_size) * ldb + i % block_size;
        pack_a_t_trans<8, T>(a_ptr, lda, b_ptr, cur_k, block_size);
    }
    for (; i + 3 < cur_m; i += 4) {
        auto a_ptr = src_a + i * lda;
        auto b_ptr = src_b + divDown(i, block_size) * ldb + i % block_size;
        pack_a_t_trans<4, T>(a_ptr, lda, b_ptr, cur_k, block_size);
    }
    for (; i + 2 < cur_m; i += 3) {
        auto a_ptr = src_a + i * lda;
        auto b_ptr = src_b + divDown(i, block_size) * ldb + i % block_size;
        pack_a_t_trans<3, T>(a_ptr, lda, b_ptr, cur_k, block_size);
    }
    for (; i + 1 < cur_m; i += 2) {
        auto a_ptr = src_a + i * lda;
        auto b_ptr = src_b + divDown(i, block_size) * ldb + i % block_size;
        pack_a_t_trans<2, T>(a_ptr, lda, b_ptr, cur_k, block_size);
    }
    for (; i < cur_m; i++) {
        auto a_ptr = src_a + i * lda;
        auto b_ptr = src_b + divDown(i, block_size) * ldb + i % block_size;
        pack_a_t_trans<1, T>(a_ptr, lda, b_ptr, cur_k, block_size);
    }
}
template void pack_col_a_t<float>(const float * a, dim_t lda, float * b, dim_t ldb, 
                   dim_t m, dim_t n, conv_gemm_config<float, float, float> &conv_gemm_conf);

} // namespace tnn
