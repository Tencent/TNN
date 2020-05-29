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

#ifdef __arm__
#ifndef __aarch64__

#include "tnn/device/arm/acc/compute/asm_func_name.S"

.text
.align 5

asm_function ConvDw3x3Bfp16SlideW 
//void ConvDw3x3Bfp16SlideW(bfp16_t *dst_z,
//                          bfp16_t **cache_line,
//                          const float* weight_z,
//                          int dst_width)

dst      .req r0
line0    .req r4
line1    .req r5
line2    .req r6
weight   .req r2
width    .req r3

w_00      .req q0
w_01      .req q1
w_02      .req q2
w_10      .req q3
w_11      .req q4
w_12      .req q5
w_20      .req q6
w_21      .req q7
w_22      .req q8

push {r4-r6, lr}

//Auto Load:
//r0:dst_z, r1:cache_line, r2:weight_z, r3: dst_width

vpush {q4-q7}

cmp width, #0
ble End

ldr r4, [r1]
ldr r5, [r1, #4]
ldr r6, [r1, #8]

vldm weight!, {d0-d15}
vld1.32 {q8}, [weight]
vld1.32 {d22}, [line0]!
vld1.32 {d24}, [line1]!
vld1.32 {d26}, [line2]!
vshll.u16  q11, d22, #16
vshll.u16  q12, d24, #16
vshll.u16  q13, d26, #16
vmul.f32 q9, q11, w_00
vmla.f32 q9, q12, w_10
vmla.f32 q9, q13, w_20
vld1.32 {d28}, [line0]!
vld1.32 {d30}, [line1]!
vld1.32 {d22}, [line2]!
vshll.u16  q14, d28, #16
vshll.u16  q15, d30, #16
vshll.u16  q11, d22, #16
vmul.f32 q10, q14, w_00
vmla.f32 q9, q14, w_01
vmla.f32 q10, q15, w_10
vmla.f32 q9, q15, w_11
vmla.f32 q10, q11, w_20
vmla.f32 q9, q11, w_21

subs width, width, #1
beq LoopDwEnd
LoopDw:
    vld1.32 {d22}, [line0]!
    vld1.32 {d24}, [line1]!
    vld1.32 {d26}, [line2]!
    vshll.u16  q11, d22, #16
    vshll.u16  q12, d24, #16
    vshll.u16  q13, d26, #16
    vmul.f32 q14, q11, w_00
    vmla.f32 q10, q11, w_01
    vmla.f32 q9, q11, w_02

    vmla.f32 q14, q12, w_10
    vmla.f32 q10, q12, w_11
    vmla.f32 q9, q12, w_12

    vmla.f32 q14, q13, w_20
    vmla.f32 q10, q13, w_21
    vmla.f32 q9, q13, w_22
    vshrn.u32  d18, q9, #16
    vst1.32 {d18}, [dst]!
    subs width, width, #1
    vmov q9, q10
    vmov q10, q14

    bne LoopDw
LoopDwEnd:

vld1.32 {d22}, [line0]!
vld1.32 {d24}, [line1]!
vld1.32 {d26}, [line2]!
vshll.u16  q11, d22, #16
vshll.u16  q12, d24, #16
vshll.u16  q13, d26, #16
vmla.f32 q9, q11, w_02
vmla.f32 q9, q12, w_12
vmla.f32 q9, q13, w_22
vshrn.u32  d18, q9, #16
vst1.32 {d18}, [dst]!

End:

vpop {q4-q7}
pop {r4-r6, pc}

#endif
#endif
