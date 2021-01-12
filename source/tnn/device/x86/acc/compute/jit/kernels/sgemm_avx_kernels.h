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

#ifndef TNN_SGEMM_AVX_KERNELS_H
#define TNN_SGEMM_AVX_KERNELS_H

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <fstream>
#include <immintrin.h>
#include <xmmintrin.h>
#include <exception>
#include <utility>

#include <xbyak/xbyak.h>

#include "tnn/device/x86/acc/compute/jit/utils/macro.h"
#include "tnn/device/x86/acc/compute/jit/common/type_def.h"
#include "tnn/device/x86/acc/compute/jit/common/abi_info.h"
#include "tnn/device/x86/acc/compute/jit/common/asm_common.h"

#include "tnn/device/x86/acc/compute/jit/kernels/base_jit_kernel.h"

#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_avx_16_i.h"
#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_avx_8_i.h"
#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_avx_4_i.h"
#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_avx_2_i.h"
#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_avx_1_i.h"

namespace TNN_NS {
namespace jit {

template<int M = 8, int N = 6, int M_BLOCK_SIZE = 16, int N_BLOCK_SIZE = 6>
class sgemm_avx_kernel: public base_jit_kernel {

public:
    static void naive_impl(const dim_t K, const float * alpha_ptr,
                           const float * src_a, const dim_t lda,
                           const float * src_b, dim_t ldb, const float * beta_ptr,
                           float * dst, dim_t ldc) {}

    using func_ptr_t = decltype(&sgemm_avx_kernel::naive_impl);

public:
    sgemm_avx_kernel() {
        switch (M) {
            case 1:
                actual = new sgemm_avx_1xi<N, M_BLOCK_SIZE, N_BLOCK_SIZE>();
                break;
            case 2:
                actual = new sgemm_avx_2xi<N, M_BLOCK_SIZE, N_BLOCK_SIZE>();
                break;
            case 4:
                actual = new sgemm_avx_4xi<N, M_BLOCK_SIZE, N_BLOCK_SIZE>();
                break;
            case 8:
                actual = new sgemm_avx_8xi<N, M_BLOCK_SIZE, N_BLOCK_SIZE>();
                break;
            case 16:
                actual = new sgemm_avx_16xi<N, M_BLOCK_SIZE, N_BLOCK_SIZE>();
                break;
            default:
                throw std::runtime_error("kernel not found for specified param."); 
                break;
        }
        ret();
    }

    virtual void * get_func_ptr() {
        if (actual != nullptr) {
            return actual->get_func_ptr();
        } else {
            throw std::runtime_error("kernel not initialized."); 
        }
        return base_jit_kernel::get_func_ptr();
    }


    virtual std::string get_kernel_name() {
        if (actual) {
            return actual->get_kernel_name();
        } else {
            throw std::runtime_error("kernel not initialized."); 
        }
        return JIT_KERNEL_NAME(sgemm_avx_kernel);
    }

    virtual size_t get_func_size() {
        if (actual) {
            return actual->get_func_size();
        } else {
            throw std::runtime_error("kernel not initialized."); 
        }
        return base_jit_kernel::get_func_size();
    }

    virtual ~sgemm_avx_kernel() {
        if (actual != nullptr) {
            delete actual;
            actual = nullptr;
        }
    }

private:
    base_jit_kernel * actual = nullptr;

};

} // namespace jit
} // namespace tnn

#endif // TNN_SGEMM_AVX_KERNELS_H
