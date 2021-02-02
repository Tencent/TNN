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

#if TNN_ARM82

#ifdef __arm__
#ifndef __aarch64__

#include "tnn/device/arm/acc/compute/asm_func_name.S"

.text
.align 5

asm_function Half2FloatKernel
//void Half2FloatKernel(float* dst, const __fp16* src, const size_t length)
//Auto Load:
//r0:dst, r1:src, r2:length
push {r4-r11, lr}
vpush {q4-q7}

cmp r2, #15
ble L16END

L16:
    sub r2, r2, #16
    vld1.16 {q0}, [r1]!
    vld1.16 {q1}, [r1]!
    cmp r2, #16
    vcvt.f32.f16 q2, d0
    vcvt.f32.f16 q3, d1
    vcvt.f32.f16 q4, d2
    vcvt.f32.f16 q5, d3
    vstm r0!, {d4-d11}
    bge L16

L16END:
cmp r2, #7
ble L8END
L8:
    sub r2, r2, #8
    vld1.16 {q0}, [r1]!
    cmp r2, #8
    vcvt.f32.f16 q2, d0
    vcvt.f32.f16 q3, d1
    vstm r0!, {d4-d7}
    bge L8

L8END:
cmp r2, #3
ble L4END
L4:
    sub r2, r2, #4
    vld1.16 d0, [r1]!
    cmp r2, #4
    vcvt.f32.f16 q2, d0
    vst1.32 {q2}, [r0]!
    bge L4

L4END:
cmp r2, #0
beq L1END

L1:
    subs r2, r2, #1
    vld1.16 d0[0], [r1]!
    vcvt.f32.f16 q2, d0
    vst1.32 d4[0], [r0]!
    bne L1
L1END:

vpop {q4-q7}
pop {r4-r11, pc}
#endif
#endif
#endif
