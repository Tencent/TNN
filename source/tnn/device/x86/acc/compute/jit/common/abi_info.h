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

#ifndef TNN_ABI_INFO_HPP_
#define TNN_ABI_INFO_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <fstream>
#include <immintrin.h>
#include <xmmintrin.h>

#include <xbyak/xbyak.h>
#include "tnn/core/macro.h"

namespace TNN_NS {
namespace jit {
namespace abi {

#ifdef XBYAK64
// ------------------------------------  64 bit related abi info ----------------------------------------

    constexpr Xbyak::Operand::Code abi_regs_callee_save[] = {
        Xbyak::Operand::RBX, Xbyak::Operand::R12,
        Xbyak::Operand::R13, Xbyak::Operand::R14, Xbyak::Operand::R15,
    #ifdef _WIN32
        Xbyak::Operand::RDI, Xbyak::Operand::RSI,
    #endif // _WIN32
    };


    constexpr Xbyak::Operand::Code abi_args_in_register[] = {
        #ifdef _WIN32
            Xbyak::Operand::RCX, Xbyak::Operand::RDX,
            Xbyak::Operand::R8,  Xbyak::Operand::R9,
        #else 
            Xbyak::Operand::RDI, Xbyak::Operand::RSI,
            Xbyak::Operand::RDX, Xbyak::Operand::RCX,
            Xbyak::Operand::R8,  Xbyak::Operand::R9,
        #endif 
    };

    constexpr const int abi_nb_args_in_register = sizeof(abi_args_in_register) / sizeof(Xbyak::Operand::Code);

    #ifdef _WIN32
    constexpr const int abi_stack_param_offset = 32;
    #else
    constexpr const int abi_stack_param_offset = 0;
    #endif

#else // XBYAK64

// ------------------------------------  32 bit related abi info ----------------------------------------

    constexpr Xbyak::Operand::Code abi_regs_callee_save[] = {
        Xbyak::Operand::EBX,  Xbyak::Operand::EDI,
        Xbyak::Operand::ESI,
    };

    static Xbyak::Operand::Code * abi_args_in_register;

    constexpr const int abi_nb_args_in_register = 0;
    constexpr const int abi_stack_param_offset = 0;
#endif // XBYAK64

constexpr const int abi_nb_regs_callee_save = sizeof(abi_regs_callee_save) / sizeof(Xbyak::Operand::Code);
constexpr const int register_width_in_bytes = sizeof(void*);


} // namespace abi
} // namespace jit
} // namespace tnn

#endif // TNN_ABI_INFO_HPP_