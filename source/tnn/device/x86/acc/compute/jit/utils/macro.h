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

#ifndef TNN_JIT_MACRO_H_
#define TNN_JIT_MACRO_H_

#ifdef XBYAK64 
#define movx movq
#define JIT_KERNEL_NAME(class)           "tnn_jit_64_"#class
#else // XBYAK64
#define movx mov
#define JIT_KERNEL_NAME(class)           "tnn_jit_32_"#class
#endif

#define MIN_(A, B) (A < B ? A : B)

#define LOOP(n, tag)                                                \
    cmp(n, 0);                                                      \
    jle("end"#tag, T_NEAR);                                         \
    L(""#tag);                                                      \
    bool _##tag##first_call = true;                                 \
    auto _##tag##condition = [&](rf_t n, bool &first_call){         \
        if (first_call) {                                           \
            first_call = false;                                     \
            return true;                                            \
        } else {                                                    \
            sub(n, 0x1);                                            \
            jne(""#tag, T_NEAR);                                    \
            L("end"#tag);                                           \
            return false;                                           \
        }                                                           \
    };                                                              \
    while(_##tag##condition(n, _##tag##first_call))

#define LOOP_STACK_VAR(n, tag)                                      \
    cmp(n, 0);                                                      \
    jle("end"#tag, T_NEAR);                                         \
    L(""#tag);                                                      \
    bool _##tag##first_call = true;                                 \
    auto _##tag##condition = [&](stack_var &n, bool &first_call){   \
        if (first_call) {                                           \
            first_call = false;                                     \
            return true;                                            \
        } else {                                                    \
            sub(n, 0x1);                                            \
            jg(""#tag, T_NEAR);                                     \
            L("end"#tag);                                           \
            return false;                                           \
        }                                                           \
    };                                                              \
    while(_##tag##condition(n, _##tag##first_call))

#define USE_REG(rf)                                                 \
    rf.aquire();                                                    \
    bool _##tag##first_call = true;                                 \
    auto _##tag##condition = [&](bool &first_call){                 \
        if (first_call) {                                           \
            first_call = false;                                     \
            return true;                                            \
        } else {                                                    \
            rf.release();                                           \
            return false;                                           \
        }                                                           \
    };                                                              \
    while(_##tag##condition(n, _##tag##first_call))


#define STACK_VAR_ARRAY_2   get_stack_var(), get_stack_var(),
#define STACK_VAR_ARRAY_3                                           \
        get_stack_var(), get_stack_var(), get_stack_var()
#define STACK_VAR_ARRAY_4 STACK_VAR_ARRAY_2 STACK_VAR_ARRAY_2
#define STACK_VAR_ARRAY_5 STACK_VAR_ARRAY_2 STACK_VAR_ARRAY_3
#define STACK_VAR_ARRAY_6 STACK_VAR_ARRAY_4 STACK_VAR_ARRAY_2
#define STACK_VAR_ARRAY_7 STACK_VAR_ARRAY_4 STACK_VAR_ARRAY_3
#define STACK_VAR_ARRAY_8 STACK_VAR_ARRAY_4 STACK_VAR_ARRAY_4

#define REG_VAR_ARRAY_2   reg_var(this), reg_var(this),
#define REG_VAR_ARRAY_3   reg_var(this), reg_var(this), reg_var(this),
#define REG_VAR_ARRAY_4   REG_VAR_ARRAY_2 REG_VAR_ARRAY_2
#define REG_VAR_ARRAY_5   REG_VAR_ARRAY_2 REG_VAR_ARRAY_3
#define REG_VAR_ARRAY_6   REG_VAR_ARRAY_2 REG_VAR_ARRAY_4
#define REG_VAR_ARRAY_7   REG_VAR_ARRAY_3 REG_VAR_ARRAY_4
#define REG_VAR_ARRAY_8   REG_VAR_ARRAY_4 REG_VAR_ARRAY_4
#define REG_VAR_ARRAY_16  REG_VAR_ARRAY_8 REG_VAR_ARRAY_8

#define VREG_VAR_ARRAY_2   vreg_var(this), vreg_var(this),
#define VREG_VAR_ARRAY_3   vreg_var(this), vreg_var(this), vreg_var(this),
#define VREG_VAR_ARRAY_4   VREG_VAR_ARRAY_2 VREG_VAR_ARRAY_2
#define VREG_VAR_ARRAY_5   VREG_VAR_ARRAY_2 VREG_VAR_ARRAY_3
#define VREG_VAR_ARRAY_6   VREG_VAR_ARRAY_2 VREG_VAR_ARRAY_4
#define VREG_VAR_ARRAY_7   VREG_VAR_ARRAY_3 VREG_VAR_ARRAY_4
#define VREG_VAR_ARRAY_8   VREG_VAR_ARRAY_4 VREG_VAR_ARRAY_4
#define VREG_VAR_ARRAY_16  VREG_VAR_ARRAY_8 VREG_VAR_ARRAY_8

#endif // TNN_JIT_MACRO_H_
