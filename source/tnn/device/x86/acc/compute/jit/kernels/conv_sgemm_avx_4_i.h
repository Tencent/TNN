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

#ifndef CONV_TNN_SGEMM_AVX_4xI_H_
#define CONV_TNN_SGEMM_AVX_4xI_H_

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

template<int I, int M_BLOCK_SIZE, int N_BLOCK_SIZE>
class conv_sgemm_avx_4xi: public base_jit_kernel {

public:
    static void naive_impl(const dim_t K,
                           const float * src_a, const dim_t lda,
                           const float * src_b, dim_t ldb,
                           float * dst, dim_t ldc,
                           const float * bias, dim_t first, dim_t act_type) {}

    using func_ptr_t = decltype(&conv_sgemm_avx_4xi::naive_impl);

    virtual std::string get_kernel_name() {
        std::stringstream buf;
        buf << JIT_KERNEL_NAME(conv_sgemm_avx_4) << "_" << I << "_" << M_BLOCK_SIZE << "_" << N_BLOCK_SIZE;
        return buf.str();
    }
    
public:
    conv_sgemm_avx_4xi() {
        constexpr int N_r = MIN_(6, I);

        declare_param<const dim_t>();       // 0. K
        declare_param<const float *>();     // 1. src_a
        declare_param<const dim_t>();       // 2. lda
        declare_param<const float *>();     // 3. src_b
        declare_param<const dim_t>();       // 4. ldb
        declare_param<float *>();           // 5. dst
        declare_param<const dim_t>();       // 6. ldc
        declare_param<const float *>();     // 7. bias
        declare_param<dim_t>();             // 8. first
        declare_param<dim_t>();             // 9. act_type

        abi_prolog();

        stack_var K         = get_arguement_to_stack(0);
        reg_var src_a       = get_arguement(1);
        reg_var lda         = get_arguement(2);
        reg_var src_b       = get_arguement(3);
        reg_var ldb         = get_arguement(4);
        reg_var dst         = get_arguement(5);
        reg_var ldc         = get_arguement(6);
        reg_var bias        = get_arguement(7);
        reg_var first       = get_arguement(8);
        reg_var act_type    = get_arguement(9);

        reg_var c[3] = {REG_VAR_ARRAY_3};
        reg_var op_6f(this);
        vreg_var v_const(this);
        vreg_var c_data[6] = {VREG_VAR_ARRAY_6};
        vreg_var a_data(this), b_data(this);
        
        ldc.restore();
        mov(c[0].aquire(), dst.restore());
        lea(c[1].aquire(), byte[dst + (ldc * 8)]);
        lea(c[2].aquire(), byte[c[1]+ (ldc * 8)]);
        dst.release();

        Xbyak::RegExp c_addr[6] = {
            Xbyak::RegExp(c[0]),
            Xbyak::RegExp(c[0] + (ldc * 4)),
            Xbyak::RegExp(c[1]),
            Xbyak::RegExp(c[1] + (ldc * 4)),
            Xbyak::RegExp(c[2]),
            Xbyak::RegExp(c[2] + (ldc * 4)),
        };

        first.restore();
        cmp(first, 0);
        jne("L_init");
        bias.restore();
        for(int i=0;i<N_r;i++) {
            c_data[i].aquire();
            vbroadcastss_sse(c_data[i].xmm(), dword[bias + i * 4]);
        }
        bias.release();
        jmp("L_init_end");
        L("L_init");
        for(int i=0;i<N_r;i++) {
            vmovups_sse(c_data[i].xmm(), xword[c_addr[i]]);
        }
        L("L_init_end");
        first.release();

        src_a.restore();
        src_b.restore();

        LOOP_STACK_VAR(K, SGEMM_AVX_8X6_K) 
        {
            a_data.aquire();
            vmovups_sse(a_data.xmm(), xword[src_a]);
            for(int i=0;i<N_r;i++) {
                b_data.aquire();
                vbroadcastss_sse(b_data.xmm(), xword[src_b + i * 4]);
                vfmadd231ps_sse(c_data[i].xmm(), a_data.xmm(), b_data.xmm());
                b_data.release();
            }
            a_data.release();
            lea(src_a, byte[src_a + M_BLOCK_SIZE * 4]);
            lea(src_b, byte[src_b + N_BLOCK_SIZE * 4]);
        }

        src_a.release();
        src_b.release();

        // only support fuse relu, relu6
        act_type.restore();
        cmp(act_type, 0);
        je("L_post_end");
            v_const.aquire();
            xorps(v_const.xmm(), v_const.xmm());
            for(int i=0;i<N_r;i++) {
                maxps(c_data[i].xmm(), v_const.xmm());
            }
            v_const.release();
        cmp(act_type, 2);
        jne("L_post_end");
            op_6f.restore();
            v_const.aquire();
            // 6.f
            mov(op_6f.cvt32(), 0x40C00000);
            movd(v_const.xmm(), op_6f.cvt32());
            shufps(v_const.xmm(), v_const.xmm(), 0);
            for(int i=0;i<N_r;i++) {
                minps(c_data[i].xmm(), v_const.xmm());
            }
            v_const.release();
            op_6f.release();
        L("L_post_end");
        act_type.release();

        for(int i=0;i<N_r;i++) {
            movups(xword[c_addr[i]], c_data[i].xmm());
        }

        abi_epilog();
        ret();
    }

    virtual ~conv_sgemm_avx_4xi() {

    }

private:

};

} // namespace jit
} // namespace tnn

#endif // CONV_TNN_SGEMM_AVX_4xI_H_
