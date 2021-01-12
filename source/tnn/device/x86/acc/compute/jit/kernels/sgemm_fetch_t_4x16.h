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

#ifndef TNN_SGEMM_FETCH_T_4x16_HPP_
#define TNN_SGEMM_FETCH_T_4x16_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <fstream>
#include <immintrin.h>
#include <xmmintrin.h>
#include <exception>
#include <utility>

#include <xbyak/xbyak.h>

#include "tnn/device/x86/acc/compute/jit/common/type_def.h"
#include "tnn/device/x86/acc/compute/jit/utils/macro.h"
#include "tnn/device/x86/acc/compute/jit/common/abi_info.h"
#include "tnn/device/x86/acc/compute/jit/common/asm_common.h"
#include "tnn/device/x86/acc/compute/jit/kernels/base_jit_kernel.h"

namespace TNN_NS {
namespace jit {

// only supports block_size >= 16
// only supports 64bit machines
class sgemm_fetch_t_4x16_ker_t: public base_jit_kernel {

public:
    static void naive_impl(const dim_t m, const float * a, const dim_t lda, float * b, const dim_t ldb, const dim_t block_size) {
    }

    using func_ptr_t = decltype(&sgemm_fetch_t_4x16_ker_t::naive_impl);

    virtual std::string get_kernel_name() {
        return JIT_KERNEL_NAME(sgemm_fetch_t_4x16);
    }

public:
    sgemm_fetch_t_4x16_ker_t() {

        declare_param<const size_t>();
        declare_param<const float *>();
        declare_param<const size_t>();
        declare_param<float *>();
        declare_param<const dim_t>();
        declare_param<const dim_t>();

        abi_prolog();

#ifdef XBYAK64

        stack_var m     = get_arguement_to_stack(0);
        stack_var a_stack = get_arguement_to_stack(1);
        reg_var   lda   = get_arguement(2);
        stack_var b_stack = get_arguement_to_stack(3);
        reg_var   ldb   = get_arguement(4);
        reg_var block_size = get_arguement(5);

        stack_var m4 = get_stack_var();
        stack_var m1 = get_stack_var();

        reg_var tmp(this);
        reg_var a_r[4] = {REG_VAR_ARRAY_4};
        reg_var b_r[4] = {REG_VAR_ARRAY_4};

        // init m4 = m / 4
        mov(tmp.aquire(), m);
        sar(tmp, 0x2);
        mov(m4, tmp);

        // init m1 = m % 4
        mov(tmp, m);
        and_(tmp, 0x3);
        mov(m1, tmp);

        lda.restore();
        lea(lda, byte[lda*4]);

        ldb.restore();
        sal(ldb, 0x6); // ldb = ldb * 16(block_size) * sizeof(float)

        block_size.restore();
        lea(block_size, byte[block_size*4]);

        // init a pointers along m 
        mov(tmp, a_stack);
        for(int i=0;i<4;i++) {
            mov(a_r[i].aquire(), tmp);
            if (i<3) add(tmp, lda);
        }

        // init b pointers along n
        mov(tmp, b_stack);
        for(int i=0;i<4;i++) {
            mov(b_r[i].aquire(), tmp);
            if (i<3) add(tmp, ldb);
        }
        tmp.release();
        ldb.release();

        vreg_var v[4][4] = {{VREG_VAR_ARRAY_4},{VREG_VAR_ARRAY_4},{VREG_VAR_ARRAY_4},{VREG_VAR_ARRAY_4}};
        
        sal(lda, 0x2);
        sal(block_size, 0x2);

        LOOP_STACK_VAR(m4, SGEMM_FETCH_T8_M8) 
        {
            for(int m=0;m<4;m++) {
                for(int n=0;n<2;n++){
                    vmovups(v[m][n].aquire(),     yword[a_r[m] + n * 32]);
                }
            }

            for(int m=0;m<4;m++) {
                vmovaps(yword[b_r[0] + (m*2 + 0) * 8 * 4],  v[m][0].release());
                vmovaps(yword[b_r[0] + (m*2 + 1) * 8 * 4],  v[m][1].release());
            }
            add(b_r[0], block_size);

            for(int m=0;m<4;m++) {
                for(int n=0;n<2;n++){
                    vmovups(v[m][n].aquire(),     yword[a_r[m] + (n + 2) * 32]);
                }
            }

            for(int m=0;m<4;m++) {
                vmovaps(yword[b_r[1] + (m*2 + 0) * 8 * 4],  v[m][0].release());
                vmovaps(yword[b_r[1] + (m*2 + 1) * 8 * 4],  v[m][1].release());
            }
            add(b_r[1], block_size);

            for(int m=0;m<4;m++) {
                for(int n=0;n<2;n++){
                    vmovups(v[m][n].aquire(),     yword[a_r[m] + (n + 4) * 32]);
                }
            }

            for(int m=0;m<4;m++) {
                vmovaps(yword[b_r[2] + (m*2 + 0) * 8 * 4],  v[m][0].release());
                vmovaps(yword[b_r[2] + (m*2 + 1) * 8 * 4],  v[m][1].release());
            }
            add(b_r[2], block_size);

            for(int m=0;m<4;m++) {
                for(int n=0;n<2;n++){
                    vmovups(v[m][n].aquire(),     yword[a_r[m] + (n + 6) * 32]);
                }
                lea(a_r[m], byte[a_r[m] + lda]);
            }

            for(int m=0;m<4;m++) {
                vmovaps(yword[b_r[3] + (m*2 + 0) * 8 * 4],  v[m][0].release());
                vmovaps(yword[b_r[3] + (m*2 + 1) * 8 * 4],  v[m][1].release());
            }
            add(b_r[3], block_size);
        }
        
        
        sar(lda, 0x2);
        sar(block_size, 0x2);

        LOOP_STACK_VAR(m1, SGEMM_FETCH_T8_M1) 
        {
            for(int n=0;n<4;n++) {
                vmovups(v[0][n].aquire(),   yword[a_r[0] + (n * 16) * 4]);
                vmovups(v[0][n+4].aquire(), yword[a_r[0] + (n * 16 + 8) * 4]);
            }

            for(int n=0;n<4;n++) {
                vmovaps(yword[b_r[n]],         v[0][n].release());
                vmovaps(yword[b_r[n] + 8 * 4], v[0][n+4].release());
            }

            add(a_r[0], lda);
            for(int i=0;i<4;i++) {
                lea(b_r[i], byte[b_r[i] + block_size]);
            }
        }

#endif
        abi_epilog();
        ret();
    }

    virtual ~sgemm_fetch_t_4x16_ker_t() {

    }

private:

};

} // namespace jit
} // namespace tnn

#endif // TNN_SGEMM_FETCH_T_4x16_HPP_