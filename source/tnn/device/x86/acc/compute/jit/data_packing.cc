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

#include "tnn/device/x86/acc/compute/jit/gemm_config.h"

namespace TNN_NS {

//  pack block_size on non-leading dimension, n denotes no-transpose. 
//  eg. input:   A MxN matrix in col major, so the storage-format is (N, M) 
//      output:  B MxN matrix in col major(N-packed), so the storage-format is 
//                    (divUp(N, block_size), M, block_size)
template<typename T>
void pack_n(const T * a, dim_t lda, T * b, dim_t ldb, dim_t m, dim_t n, gemm_config<T, T, T> &gemm_conf)
{
    int block_size = gemm_conf.n_block_;
    dim_t i=0; 
    for(;n-i>= block_size;i+= block_size) {
        const T * cur_a = a + i * lda;
        T * cur_b = b + i * ldb;
        gemm_conf.pack_n_ker_[block_size](m, cur_a, lda, cur_b, ldb, block_size);
    }
    dim_t tail = n - i;
    if (tail > 0) {
        const T * cur_a = a + i * lda;
        T * cur_b = b + i * ldb;
        gemm_conf.pack_n_ker_[tail](m, cur_a, lda, cur_b, ldb, block_size);
    }

}

template 
void pack_n<float>(const float * a, dim_t lda, float * b, dim_t ldb, 
                   dim_t m, dim_t n,
                   gemm_config<float, float, float> &gemm_conf);

//  pack block_size on leading dimension, t denotes transpose. 
//  eg. input:   A MxN matrix in row major, so the storage-format is (M, N) 
//      output:  B MxN matrix in col major(N-packed), so the storage-format is 
//                    (divUp(N, block_size), M, block_size)
template<typename T>
void pack_t(const T * a, dim_t lda, T * b, dim_t ldb, dim_t m, dim_t n, gemm_config<T, T, T> &gemm_conf) 
{
    dim_t block_size = gemm_conf.m_block_;
    dim_t i = 0; 

    if (block_size == 16) {
        for(;i + 64 <=n;i+=64) {
            const T * cur_a = a + i;
            T * cur_b = b + i * ldb;
            gemm_conf.pack_t_4x16_ker_(m, cur_a, lda, cur_b, ldb, block_size);
        }
    } else if (block_size == 8) {
        
    }

    for(;i<n;) {
        const T * cur_a = a + i;
        T * cur_b = b + divDown(i, block_size) * ldb + i % block_size;
        dim_t cur_n = std::min(n - i, block_size);
        switch(cur_n) {
            case 1:
                gemm_conf.pack_t_ker_[1](m, cur_a, lda, cur_b, ldb, block_size);
                i+=1;
                break;
            case 2:
                gemm_conf.pack_t_ker_[2](m, cur_a, lda, cur_b, ldb, block_size);
                i+=2;
                break;
            case 3:
                gemm_conf.pack_t_ker_[3](m, cur_a, lda, cur_b, ldb, block_size);
                i+=3;
                break;
            case 4:
            case 5:
            case 6:
            case 7:
                gemm_conf.pack_t_ker_[4](m, cur_a, lda, cur_b, ldb, block_size);
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
                gemm_conf.pack_t_ker_[8](m, cur_a, lda, cur_b, ldb, block_size);
                i+=8;
                break;
            default:
                gemm_conf.pack_t_ker_[16](m, cur_a, lda, cur_b, ldb, block_size);
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
void pack_t<float>(const float * a, dim_t lda, float * b, dim_t ldb, 
                   dim_t m, dim_t n,
                   gemm_config<float, float, float> &gemm_conf);

} // namespace tnn
