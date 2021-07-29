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

#ifndef TNN_SGEMM_FETCH_T_16_HPP_
#define TNN_SGEMM_FETCH_T_16_HPP_

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

class sgemm_fetch_t_16_ker_t: public base_jit_kernel {

public:
    static void naive_impl(const dim_t m, const float * a, const dim_t lda, float * b, const dim_t ldb, const dim_t block_size) {
    }

    using func_ptr_t = decltype(&sgemm_fetch_t_16_ker_t::naive_impl);

    virtual std::string get_kernel_name() {
        return JIT_KERNEL_NAME(sgemm_fetch_t_16);
    }

public:
    sgemm_fetch_t_16_ker_t() {

        declare_param<const size_t>();
        declare_param<const float *>();
        declare_param<const size_t>();
        declare_param<float *>();
        declare_param<const dim_t>();
        declare_param<const dim_t>();

        abi_prolog();


        stack_var m     = get_arguement_to_stack(0);
        stack_var a_stack = get_arguement_to_stack(1);
        stack_var lda   = get_arguement_to_stack(2);
        reg_var   b_ptr = get_arguement(3);
        stack_var ldb   = get_arguement_to_stack(4);
        stack_var block_size = get_arguement_to_stack(5);

        stack_var m4 = get_stack_var();
        stack_var m1 = get_stack_var();

        reg_var tmp(this);
        reg_var a_r[4] = {REG_VAR_ARRAY_4};

        reg_var ldax4(this);
        reg_var block_size_x4(this);

        // init m4 = m / 4
        mov(tmp.aquire(), m);
        sar(tmp, 0x2);
        mov(m4, tmp);

        // init m1 = m % 4
        mov(tmp, m);
        and_(tmp, 0x3);
        mov(m1, tmp);

        mov(ldax4.aquire(), lda);
        lea(ldax4, qword[ldax4*4]);
        ldax4.stash();

        // init a pointers 
        mov(tmp, a_stack);
        for(int i=0;i<4;i++) {
            mov(a_r[i].aquire(), tmp);
            if (i<3) add(tmp, ldax4);
        }
        tmp.release();

        mov(block_size_x4.aquire(), block_size);
        lea(block_size_x4, qword[block_size_x4 * 4]);
        block_size_x4.stash();

        vreg_var v[8] = {VREG_VAR_ARRAY_8};
        
        LOOP_STACK_VAR(m4, SGEMM_FETCH_T8_M8) 
        {
            //read 
            ldax4.restore();
            for(int i=0;i<4;i++) {
                vmovups(v[i].aquire(),   yword[a_r[i]]);
                vmovups(v[i+4].aquire(), yword[a_r[i] + 8 * 4]);
                lea(a_r[i], byte[a_r[i] + ldax4 * 4]); // next 4 lines
            }
            ldax4.release();

            // write
            b_ptr.restore();
            block_size_x4.restore();
            for(int i=0;i<4;i++) {
                vmovups(yword[b_ptr], v[i].release());
                vmovups(yword[b_ptr + 8 * 4], v[i+4].release());
                lea(b_ptr, byte[b_ptr + block_size_x4]);
            }
            b_ptr.stash();
            block_size_x4.release();
        }
        
        LOOP_STACK_VAR(m1, SGEMM_FETCH_T8_M1) 
        {
            //read 
            vmovups(v[0].aquire(), yword[a_r[0]]);
            vmovups(v[1].aquire(), yword[a_r[0] + 8 * 4]);

            lea(a_r[0], byte[a_r[0] + ldax4.restore()]);
            ldax4.release();

            vmovups(yword[b_ptr.restore()], v[0].release());
            vmovups(yword[b_ptr + 8 * 4], v[1].release());
            b_ptr.release();

            add(b_ptr.v_stack_, block_size_x4.restore());
            block_size_x4.release();
        }

        abi_epilog();
        ret();
    }

    virtual ~sgemm_fetch_t_16_ker_t() {

    }

private:

};

} // namespace jit
} // namespace tnn

#endif // TNN_SGEMM_FETCH_T_16_HPP_