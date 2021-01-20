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

#include "tnn/device/x86/acc/compute/jit/utils/cpu_isa.h"

#include <stdio.h>

#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>

namespace TNN_NS {

using namespace Xbyak::util;

static Xbyak::util::Cpu cpu;

bool cpu_with_isa(x86_isa_t arch) {
    switch (arch) {
        case sse42:
            return cpu.has(Cpu::tSSE42);
        case avx:
            return cpu.has(Cpu::tAVX);
        case avx2:
            return cpu.has(Cpu::tAVX2) && cpu.has(Cpu::tFMA);
        case avx512:
            return cpu.has(Cpu::tAVX512F)  && cpu.has(Cpu::tAVX512BW) &&
                   cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);
        case avx512_vnni:
            return cpu.has(Cpu::tAVX512F)  && cpu.has(Cpu::tAVX512BW) &&
                   cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ) &&
                   cpu.has(Cpu::tAVX512_VNNI);
        default:
            return false;
    }
    return false;
}

}
