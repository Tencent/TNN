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

#ifndef TNN_SGEMM_FETCH_T_I_HPP_
#define TNN_SGEMM_FETCH_T_I_HPP_

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

template<int I>
class sgemm_fetch_t_i_ker_t: public base_jit_kernel {

public:
    static void naive_impl(const dim_t m, const float * a, const dim_t lda, float * b, const dim_t ldb, const dim_t block_size) {
    }

    using func_ptr_t = decltype(&sgemm_fetch_t_i_ker_t::naive_impl);

    virtual std::string get_kernel_name() {
        std::stringstream buf;
        buf << JIT_KERNEL_NAME(sgemm_fetch_t_) << I;
        return buf.str();
    }

public:
    sgemm_fetch_t_i_ker_t() {

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
        stack_var b_stack = get_arguement_to_stack(3);
        stack_var ldb   = get_arguement_to_stack(4);
        stack_var block_size = get_arguement_to_stack(5);

        stack_var m4 = get_stack_var();
        stack_var m1 = get_stack_var();

        reg_var tmp(this);
        reg_var a_r[2] = {REG_VAR_ARRAY_2};
        reg_var b_r(this);

        reg_var ldax4(this);
        reg_var ldax8(this);
        reg_var block_size_x4(this);

        // init m8 = m / 4
        mov(tmp.aquire(), m);
        sar(tmp, 0x2);
        mov(m4, tmp);

        // init m1 = m % 4
        mov(tmp, m);
        and_(tmp, 0x3);
        mov(m1, tmp);

        mov(b_r.aquire(), b_stack);
        b_r.stash();

        mov(ldax4.aquire(), lda);
        lea(ldax4, qword[ldax4*sizeof(float)]);
        ldax4.store();

        // init a pointers 
        mov(tmp, a_stack);
        mov(a_r[0].aquire(), tmp);
        add(tmp, ldax4);
        mov(a_r[1].aquire(), tmp);
        tmp.release();

        // lda for 4 lines
        lea(ldax8.aquire(), qword[ldax4.release()*2]);
        ldax8.stash();

        mov(block_size_x4.aquire(), block_size);
        lea(block_size_x4, qword[block_size_x4*sizeof(float)]);
        block_size_x4.stash();

        stack_var a_data[4][8] = {
            {STACK_VAR_ARRAY_8},
            {STACK_VAR_ARRAY_8},
            {STACK_VAR_ARRAY_8},
            {STACK_VAR_ARRAY_8},
        };

        LOOP_STACK_VAR(m4, SGEMM_FETCH_TI_M4) 
        {
            // load 
            ldax8.restore();
            Xbyak::RegExp a_addr[4] = {
                Xbyak::RegExp(a_r[0]),
                Xbyak::RegExp(a_r[1]),
                Xbyak::RegExp(a_r[0] + ldax8),
                Xbyak::RegExp(a_r[1] + ldax8),
            };

            for(int line=0;line<4;line++) {
                for(int i=0;i<I;i++) {
                    mov(tmp.aquire().cvt32(), dword[a_addr[line] + i * sizeof(float)]);
                    mov(a_data[line][i], tmp.release().cvt32());
                }
            }
            lea(a_r[0], byte[a_r[0] + ldax8 * 2]);
            lea(a_r[1], byte[a_r[1] + ldax8 * 2]);

            ldax8.release();

            // store
            b_r.restore();
            for(int line=0;line<4;line++) {
                for(int i=0;i<I;i++) {
                    mov(tmp.aquire().cvt32(), a_data[line][i]);
                    mov(dword[b_r + i * sizeof(float)], tmp.release().cvt32());
                }
                lea(b_r, byte[b_r + block_size_x4.restore()]);
                block_size_x4.release();
            }
            b_r.stash();
        }
        
        LOOP_STACK_VAR(m1, SGEMM_FETCH_TI_M1) 
        {
            for(int i=0;i<I;i++) {
                mov(tmp.aquire().cvt32(), dword[a_r[0] + i * sizeof(float)]);
                mov(a_data[0][i], tmp.release().cvt32());
            }
            lea(a_r[0], byte[a_r[0] + ldax4.restore()]);
            ldax4.release();

            b_r.restore();
            for(int i=0;i<I;i++) {
                mov(tmp.aquire().cvt32(), a_data[0][i]);
                mov(dword[b_r + i * sizeof(float)], tmp.release().cvt32());
            }
            lea(b_r, byte[b_r + block_size_x4.restore()]);
            b_r.stash();
        }

        abi_epilog();
        ret();
    }

    virtual ~sgemm_fetch_t_i_ker_t() {

    }

private:

};

class sgemm_fetch_t_1_ker_t: public sgemm_fetch_t_i_ker_t<1> {};
class sgemm_fetch_t_2_ker_t: public sgemm_fetch_t_i_ker_t<2> {};
class sgemm_fetch_t_3_ker_t: public sgemm_fetch_t_i_ker_t<3> {};
// class sgemm_fetch_t_4_ker_t: public sgemm_fetch_t_i_ker_t<4> {};
class sgemm_fetch_t_5_ker_t: public sgemm_fetch_t_i_ker_t<5> {};
class sgemm_fetch_t_6_ker_t: public sgemm_fetch_t_i_ker_t<6> {};
class sgemm_fetch_t_7_ker_t: public sgemm_fetch_t_i_ker_t<7> {};

} // namespace jit
} // namespace tnn

#endif // TNN_SGEMM_FETCH_T_I_HPP_