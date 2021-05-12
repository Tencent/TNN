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

#ifndef TNN_SGEMM_FETCH_N_BASE_HPP_
#define TNN_SGEMM_FETCH_N_BASE_HPP_

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
#include "tnn/device/x86/acc/compute/jit/common/abi_info.h"
#include "tnn/device/x86/acc/compute/jit/common/asm_common.h"
#include "tnn/device/x86/acc/compute/jit/kernels/base_jit_kernel.h"

namespace TNN_NS {
namespace jit {

//  only support block_size = 6
template<int I>
class sgemm_fetch_n_i_ker_t: public base_jit_kernel {

public:
    static void naive_impl(const dim_t m, const float * src, const dim_t lda, float * dst, const dim_t ldb, const dim_t block_size) {
    }

    using func_ptr_t = decltype(&sgemm_fetch_n_i_ker_t::naive_impl);

    virtual std::string get_kernel_name() {
        std::stringstream buf;
        buf << JIT_KERNEL_NAME(sgemm_fetch_n_) << I;
        return buf.str();
    }

public:
    sgemm_fetch_n_i_ker_t() {
        constexpr int I4 = MIN_(I, 4);
        constexpr int I6 = MIN_(I, 6);

        declare_param<const size_t>();
        declare_param<const float *>();
        declare_param<const size_t>();
        declare_param<float *>();
        declare_param<const size_t>();
        declare_param<const size_t>();

        abi_prolog();


        stack_var m         = get_arguement_to_stack(0);
        stack_var a_stack   = get_arguement_to_stack(1);
        stack_var lda       = get_arguement_to_stack(2);
        stack_var b_stack   = get_arguement_to_stack(3);
        stack_var ldb       = get_arguement_to_stack(4);
        reg_var   block_size= get_arguement(5);

        stack_var m8 = get_stack_var();
        stack_var m1 = get_stack_var();

        reg_var tmp(this);
        reg_var a_r[6] = {REG_VAR_ARRAY_6};
        stack_var a_s[6] = {STACK_VAR_ARRAY_6};

        reg_var ldax4(this);

        // init m8 = m / 8
        mov(tmp.aquire(), m);
        sar(tmp, 0x3);
        mov(m8, tmp);

        // init m1 = m % 8
        mov(tmp, m);
        and_(tmp, 0x7);
        mov(m1, tmp);

        block_size.restore();
        cmp(block_size, 0x6);
        block_size.release();
        jne("FUNCTION_END", T_NEAR);

        // init ldax4 = lda * sizeof(float);
        mov(ldax4.aquire(), lda);
        lea(ldax4, qword[ldax4*sizeof(float)]);


        // init a pointers 
        mov(tmp, a_stack);
        mov(a_r[0].aquire(), tmp);

        add(tmp, ldax4);
        mov(a_r[1].aquire(), tmp);

        add(tmp, ldax4);
        mov(a_r[2].aquire(), tmp);

        add(tmp, ldax4);
        mov(a_r[3].aquire(), tmp);

        add(tmp, ldax4);
        mov(a_s[4], tmp);

        add(tmp, ldax4.release());
        mov(a_s[5], tmp);

        reg_var b_r(this);

        LOOP_STACK_VAR(m8, SGEMM_FETCH_N6_M8) 
        {
            stack_var a0_data[8] = {STACK_VAR_ARRAY_8};
            stack_var a1_data[8] = {STACK_VAR_ARRAY_8};
            stack_var a2_data[8] = {STACK_VAR_ARRAY_8};
            stack_var a3_data[8] = {STACK_VAR_ARRAY_8};
            stack_var a4_data[8] = {STACK_VAR_ARRAY_8};
            stack_var a5_data[8] = {STACK_VAR_ARRAY_8};
            stack_var * a_data[] = {
                a0_data, a1_data, a2_data,
                a3_data, a4_data, a5_data, 
            };

            // ai in register
            for(int i=0;i<I4;i++) {
                for(int j=0;j<8;j++) {
                    mov(tmp.cvt32(), dword[a_r[i] + j * sizeof(float)]);
                    mov(a_data[i][j], tmp.cvt32());
                }
            }

            // ai in stack
            reg_var a_r_tmp(this);
            for(int i=4;i<I6;i++) {
                for(int j=0;j<8;j++) {
                    mov(a_r_tmp.aquire(), a_s[i]);
                    mov(tmp.cvt32(), dword[a_r_tmp.release() + j * sizeof(float)]);
                    mov(a_data[i][j], tmp.cvt32());
                }
            }


            mov(b_r.aquire(), b_stack);

            for(int j=0;j<8;j++) {
                for(int i=0;i<I6;i++) {
                    mov(tmp.cvt32(), a_data[i][j]);
                    mov(dword[b_r + (j * 6 + i) * sizeof(float)], tmp.cvt32());
                }
            }

            b_r.release();

            size_t vsize_in_bytes = 8 * sizeof(float);

            add(b_stack, 6 * vsize_in_bytes);
            add(a_r[0], vsize_in_bytes);
            add(a_r[1], vsize_in_bytes);
            add(a_r[2], vsize_in_bytes);
            add(a_r[3], vsize_in_bytes);
            add(a_s[4], vsize_in_bytes);
            add(a_s[5], vsize_in_bytes);
        }

        size_t ele_size = sizeof(float);
        mov(b_r.aquire(), b_stack);
        LOOP_STACK_VAR(m1, SGEMM_FETCH_N6_M1) 
        {
            for(int i=0;i<I4;i++) {
                mov(tmp.cvt32(), dword[a_r[i]]);
                mov(dword[b_r + i * ele_size ], tmp.cvt32());
                add(a_r[i], sizeof(float));
            }

            for(int i=4;i<I6;i++) {
                mov(tmp, a_s[i]);
                add(a_s[i], sizeof(float));
                mov(tmp.cvt32(), dword[tmp]);
                mov(dword[b_r + i * ele_size ], tmp.cvt32());
            }

            add(b_r, 6 * sizeof(float));
        }

        L("FUNCTION_END");
        abi_epilog();
        ret();
    }

    virtual ~sgemm_fetch_n_i_ker_t() {

    }

private:

};


class sgemm_fetch_n_1_ker_t: public sgemm_fetch_n_i_ker_t<1> {};
class sgemm_fetch_n_2_ker_t: public sgemm_fetch_n_i_ker_t<2> {};
class sgemm_fetch_n_3_ker_t: public sgemm_fetch_n_i_ker_t<3> {};
class sgemm_fetch_n_4_ker_t: public sgemm_fetch_n_i_ker_t<4> {};
class sgemm_fetch_n_5_ker_t: public sgemm_fetch_n_i_ker_t<5> {};
class sgemm_fetch_n_6i_ker_t: public sgemm_fetch_n_i_ker_t<6> {};


} // namespace jit
} // namespace tnn

#endif // TNN_SGEMM_FETCH_N_BASE_HPP_