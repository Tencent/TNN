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

asm_function ConvDw5x5FloatSlideW
//void ConvDw5x5FloatSlideW(float *dst_z,
//                        float **cache_line,
//                        const float* weight_z,
//                        int dst_width)

dst      .req r0
line0    .req r4
line1    .req r5
line2    .req r6
line3    .req r7
line4    .req r8
weight   .req r2
width    .req r3

push {r4-r8, lr}

//Auto Load:
//r0:dst_z, r1:cache_line, r2:weight_z, r3: dst_width

vpush {q4-q7}

cmp width, #0
ble End

ldr r4, [r1]
ldr r5, [r1, #4]
ldr r6, [r1, #8]
ldr r7, [r1, #12]
ldr r8, [r1, #16]

L4:
cmp width, #3
ble L1

vld1.32 {d0-d3}, [weight]!
vld1.32 {d4-d7}, [weight]!
vld1.32 {d8-d11}, [line0]!
vld1.32 {d12-d15}, [line0]!

    L4Loop:
    //  line0
    vld1.32  {d16-d19},  [line0]!
    vmul.f32 q12,  q4,   q0
    vmul.f32 q13,  q5,   q0
    vmul.f32 q14,  q6,   q0
    vmul.f32 q15,  q7,   q0
    vld1.32  {d20-d23},  [line0]!
    sub  line0, line0, #64
    vmla.f32 q12,  q5,   q1
    vmla.f32 q13,  q6,   q1
    vmla.f32 q14,  q7,   q1
    vmla.f32 q15,  q8,   q1
    vld1.32  {d8-d11},   [line1]!
    vmla.f32 q12,  q6,   q2
    vmla.f32 q13,  q7,   q2
    vmla.f32 q14,  q8,   q2
    vmla.f32 q15,  q9,   q2
    vld1.32   {d0-d3},   [weight]!
    vmla.f32 q12,  q7,   q3
    vmla.f32 q13,  q8,   q3
    vmla.f32 q14,  q9,   q3
    vmla.f32 q15,  q10,  q3
    vld1.32  {d12-d15},  [line1]!
    vmla.f32 q12,  q8,   q0
    vmla.f32 q13,  q9,   q0
    vmla.f32 q14,  q10,  q0
    vmla.f32 q15,  q11,  q0
    vld1.32   {d4-d7},   [weight]!
    //  line1
    vmla.f32 q12,  q4,   q1
    vmla.f32 q13,  q5,   q1
    vmla.f32 q14,  q6,   q1
    vmla.f32 q15,  q7,   q1
    vld1.32  {d16-d19},  [line1]!
    vmla.f32 q12,  q5,   q2
    vmla.f32 q13,  q6,   q2
    vmla.f32 q14,  q7,   q2
    vmla.f32 q15,  q8,   q2
    vld1.32   {d0-d3},   [weight]!
    vmla.f32 q12,  q6,   q3
    vmla.f32 q13,  q7,   q3
    vld1.32  {d20-d23},  [line1]!
    vmla.f32 q14,  q8,   q3
    vmla.f32 q15,  q9,   q3
    sub  line1, line1, #64
    vmla.f32 q12,  q7,   q0
    vmla.f32 q13,  q8,   q0
    vld1.32  {d8-d11},   [line2]!
    vmla.f32 q14,  q9,   q0
    vmla.f32 q15,  q10,  q0
    vld1.32  {d4-d7},    [weight]!
    vmla.f32 q12,  q8,   q1
    vmla.f32 q13,  q9,   q1
    vld1.32  {d12-d15},  [line2]!
    vmla.f32 q14,  q10,  q1
    vmla.f32 q15,  q11,  q1
    //  line2
    vmla.f32 q12,  q4,   q2
    vmla.f32 q13,  q5,   q2
    vld1.32  {d16-d19},  [line2]!
    vmla.f32 q14,  q6,   q2
    vmla.f32 q15,  q7,   q2
    vld1.32  {d0-d3},    [weight]!
    vmla.f32 q12,  q5,   q3
    vmla.f32 q13,  q6,   q3
    vld1.32  {d20-d23},  [line2]!
    vmla.f32 q14,  q7,   q3
    vmla.f32 q15,  q8,   q3
    vld1.32  {d4-d7},    [weight]!
    sub  line2, line2, #64
    vmla.f32 q12,  q6,   q0
    vmla.f32 q13,  q7,   q0
    vmla.f32 q14,  q8,   q0
    vmla.f32 q15,  q9,   q0
    vld1.32  {d8-d11},   [line3]!
    vmla.f32 q12,  q7,   q1
    vmla.f32 q13,  q8,   q1
    vmla.f32 q14,  q9,   q1
    vmla.f32 q15,  q10,  q1
    vld1.32  {d0-d3},    [weight]!
    vmla.f32 q12,  q8,   q2
    vmla.f32 q13,  q9,   q2
    vld1.32  {d12-d15},  [line3]!
    vmla.f32 q14,  q10,  q2
    vmla.f32 q15,  q11,  q2
    //  line3
    vmla.f32 q12,  q4,   q3
    vmla.f32 q13,  q5,   q3
    vld1.32  {d16-d19},  [line3]!
    vmla.f32 q14,  q6,   q3
    vmla.f32 q15,  q7,   q3
    vld1.32  {d4-d7},    [weight]!
    vmla.f32 q12,  q5,   q0
    vmla.f32 q13,  q6,   q0
    vld1.32  {d20-d23},  [line3]!
    vmla.f32 q14,  q7,   q0
    vmla.f32 q15,  q8,   q0
    sub  line3, line3, #64
    vmla.f32 q12,  q6,   q1
    vmla.f32 q13,  q7,   q1
    vmla.f32 q14,  q8,   q1
    vmla.f32 q15,  q9,   q1
    vld1.32  {d0-d3},    [weight]!
    vmla.f32 q12,  q7,   q2
    vmla.f32 q13,  q8,   q2
    vmla.f32 q14,  q9,   q2
    vmla.f32 q15,  q10,  q2
    vld1.32  {d8-d11},  [line4]!
    vmla.f32 q12,  q8,   q3
    vmla.f32 q13,  q9,   q3
    vld1.32  {d12-d15},  [line4]!
    vmla.f32 q14,  q10,  q3
    vmla.f32 q15,  q11,  q3
    //  line4
    vmla.f32 q12,  q4,   q0
    vmla.f32 q13,  q5,   q0
    vld1.32  {d16-d19},  [line4]!
    vmla.f32 q14,  q6,   q0
    vmla.f32 q15,  q7,   q0
    vld1.32  {d4-d7},    [weight]!
    vmla.f32 q12,  q5,   q1
    vmla.f32 q13,  q6,   q1
    vmla.f32 q14,  q7,   q1
    vmla.f32 q15,  q8,   q1
    vld1.32  {d20-d23},  [line4]!
    vmla.f32 q12,  q6,   q2
    vmla.f32 q13,  q7,   q2
    vmla.f32 q14,  q8,   q2
    vmla.f32 q15,  q9,   q2
    vld1.32  {d4-d5},    [weight]!
    sub  line4, line4, #64
    vmla.f32 q12,  q7,   q3
    vmla.f32 q13,  q8,   q3
    vld1.32  {d8-d11},   [line0]!
    sub  weight,  weight, #400
    vmla.f32 q14,  q9,   q3
    vmla.f32 q15,  q10,  q3
    vld1.32  {d0-d3},    [weight]!
    vmla.f32 q12,  q8,   q2
    vmla.f32 q13,  q9,   q2
    vld1.32  {d12-d15},  [line0]!
    vmla.f32 q14,  q10,  q2
    vmla.f32 q15,  q11,  q2
    vst1.32  {d24-d27},  [dst]!
    vld1.32  {d4-d7},    [weight]!
    sub  width,   width, #4
    vst1.32  {d28-d31},  [dst]!
    cmp width, #4
    bge L4Loop

sub weight, weight, #64
sub line0, line0, #64

L1:
cmp width, #0
ble End

    L1Loop:
    vldm weight!, {d0-d9}
    vldm line0, {d10-d19}
    vmul.f32 q12, q5, q0
    vmul.f32 q13, q6, q1
    vmla.f32 q12, q7, q2
    vmla.f32 q13, q8, q3
    vmla.f32 q12, q9, q4
    add line0, line0, #16

    vldm weight!, {d0-d9}
    vldm line1, {d10-d19}
    vmla.f32 q12, q5, q0
    vmla.f32 q13, q6, q1
    vmla.f32 q12, q7, q2
    vmla.f32 q13, q8, q3
    vmla.f32 q12, q9, q4
    add line1, line1, #16

    vldm weight!, {d0-d9}
    vldm line2, {d10-d19}
    vmla.f32 q12, q5, q0
    vmla.f32 q13, q6, q1
    vmla.f32 q12, q7, q2
    vmla.f32 q13, q8, q3
    vmla.f32 q12, q9, q4
    add line2, line2, #16

    vldm weight!, {d0-d9}
    vldm line3, {d10-d19}
    vmla.f32 q12, q5, q0
    vmla.f32 q13, q6, q1
    vmla.f32 q12, q7, q2
    vmla.f32 q13, q8, q3
    vmla.f32 q12, q9, q4
    add line3, line3, #16

    vldm weight!, {d0-d9}
    vldm line4, {d10-d19}
    vmla.f32 q12, q5, q0
    vmla.f32 q13, q6, q1
    vmla.f32 q12, q7, q2
    vmla.f32 q13, q8, q3
    vmla.f32 q12, q9, q4
    add line4, line4, #16
    sub weight, weight, #400

    vadd.f32 q12, q12, q13
    sub width, width, #1
    vst1.32 {d24-d25}, [dst]!
    cmp width, #1
    bge L1Loop

End:

vpop {q4-q7}
pop {r4-r8, pc}

#endif
#endif
