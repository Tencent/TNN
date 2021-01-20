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

#ifndef TNN_ASM_COMMON_HPP_
#define TNN_ASM_COMMON_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <fstream>
#include <vector>
#include <immintrin.h>
#include <xmmintrin.h>

#include <xbyak/xbyak.h>
#include "tnn/core/macro.h"

namespace TNN_NS {
namespace jit {
namespace common {

#ifdef XBYAK64
// ------------------------------------  64 bit related ----------------------------------------

typedef Xbyak::Reg64 rf_;
#define rword_ Xbyak::util::qword

static rf_ bp(Xbyak::Operand::RBP);
static rf_ sp(Xbyak::Operand::RSP);

const std::vector<Xbyak::Reg64> regs_= {
    Xbyak::util::rax, Xbyak::util::rbx, Xbyak::util::rcx, Xbyak::util::rdx,
    /*Xbyak::util::rsp, Xbyak::util::rbp,*/ Xbyak::util::rsi, Xbyak::util::rdi,
    Xbyak::util::r8 , Xbyak::util::r9 , Xbyak::util::r10, Xbyak::util::r11,
    Xbyak::util::r12, Xbyak::util::r13, Xbyak::util::r14, Xbyak::util::r15,
};

#else // XBYAK64
// ------------------------------------  32 bit related ----------------------------------------

typedef Xbyak::Reg32 rf_;
#define rword_ Xbyak::util::dword

static rf_ bp(Xbyak::Operand::EBP);
static rf_ sp(Xbyak::Operand::ESP);

const std::vector<Xbyak::Reg32> regs_= {
    Xbyak::util::eax, Xbyak::util::ebx, Xbyak::util::ecx, Xbyak::util::edx,
    /*Xbyak::util::esp, Xbyak::util::ebp,*/ Xbyak::util::esi, Xbyak::util::edi,
};

#endif // XBYAK64

const std::vector<Xbyak::Mmx> mmx_ = {
    Xbyak::Mmx(0), Xbyak::Mmx(1), Xbyak::Mmx(2), Xbyak::Mmx(3),
    Xbyak::Mmx(4), Xbyak::Mmx(5), Xbyak::Mmx(6), Xbyak::Mmx(7),
#ifdef XBYAK64
    Xbyak::Mmx(8), Xbyak::Mmx(9), Xbyak::Mmx(10),Xbyak::Mmx(11),
    Xbyak::Mmx(12),Xbyak::Mmx(13),Xbyak::Mmx(14),Xbyak::Mmx(15)
#endif
};

// const std::vector<Xbyak::Xmm> xmm_ = {
//     Xbyak::Xmm(0), Xbyak::Xmm(1), Xbyak::Xmm(2), Xbyak::Xmm(3),
//     Xbyak::Xmm(4), Xbyak::Xmm(5), Xbyak::Xmm(6), Xbyak::Xmm(7),
// #ifdef XBYAK64
//     Xbyak::Xmm(8), Xbyak::Xmm(9), Xbyak::Xmm(10),Xbyak::Xmm(11),
//     Xbyak::Xmm(12),Xbyak::Xmm(13),Xbyak::Xmm(14),Xbyak::Xmm(15)
// #endif
// };

// const std::vector<Xbyak::Ymm> ymm_ = {
//     Xbyak::Ymm(0), Xbyak::Ymm(1), Xbyak::Ymm(2), Xbyak::Ymm(3),
//     Xbyak::Ymm(4), Xbyak::Ymm(5), Xbyak::Ymm(6), Xbyak::Ymm(7),
// #ifdef XBYAK64
//     Xbyak::Ymm(8), Xbyak::Ymm(9), Xbyak::Ymm(10),Xbyak::Ymm(11),
//     Xbyak::Ymm(12),Xbyak::Ymm(13),Xbyak::Ymm(14),Xbyak::Ymm(15)
// #endif
// };

} // namespace common
} // namespace jit
} // namespace tnn

#endif // TNN_ASM_COMMON_HPP_