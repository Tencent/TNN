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

asm_function DepthwiseI8K3S1Kernel 

//void DepthwiseI8K3S1Kernel(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z,
//                           long src_y_step, long src_w_step, long dst_depth, const float* scale_z, long dx, long dc)

dst             .req r0
src             .req r1
weight          .req r2
bias_z          .req r3
src_y_step      .req r4
src_w_step      .req r5
dst_depth       .req r6
scale_z         .req r7
dx              .req r8
dc              .req r9

push {r4, r5, r6, r7, r8, r9, r10, r11, lr}

//Auto Load:
//r0:dst, r1:src, r2:weight, r3:bias_z

//Load from sp
//r4:src_y_step, r5:src_w_step, r6:dst_depth, r7:scale_z, r8:dx, r9:dc
ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
ldr r7, [sp, #48]
ldr r8, [sp, #52]
ldr r9, [sp, #56]

vpush {q4-q7}
sub   sp, sp, #16

lsl     r10, dc,  #2
add     r11, r10, #16
add     r12, bias_z, r11
add     r3,  bias_z, r10
vld1.32 {d20-d21}, [r12]
vld1.32 {d4-d5},   [r3]
vmov    q5,  q10
vmov    q1,  q10
vmov    q0,  q10
vmov    q4,  q2
vmov    q8,  q2
vmov    q9,  q2
mov     r14, #3

mla     r3,  dx, src_w_step, dc
mla     r11, dx, dst_depth,  dc
add     weight, weight, dc
add     src,    src,    r3

.L2:
    mov     r3, src
    vld1.8  {d22}, [r3], dst_depth
    vld1.8  {d30}, [r3], dst_depth
    vld1.8  {d26}, [weight], dst_depth
    vld1.8  {d6},  [r3], dst_depth
    vmovl.s8        q13, d26            // b0
    vmovl.s8        q3,  d6             // a2
    vmovl.s8        q15, d30            // a1
    vld1.8  {d12}, [r3], dst_depth
    vstr    d22, [sp]
    vmovl.s8        q6, d12             // a3
    vmov    d22, d27    // b0 h
    vmov    d27, d6     // a2 l
    vmov    d6,  d30    // a1 l
    vldr    d30, [sp]
    vmov    d29, d12    // a3 l
    vmov    d28, d13    // a3 h
    vld1.8  {d14}, [weight], dst_depth
    vmovl.s8        q6, d30             // a0
    vld1.8  {d24}, [r3], dst_depth
    vmov    d23, d26    // b0 l
    vmovl.s8        q12, d24            // a4
    vmovl.s8        q7, d14             // b1
    vmov    d26, d7     // a2 h
    vmlal.s16       q10, d13, d22       // a0 h, b0 h -> acc01
    vld1.8  {d7}, [weight], dst_depth
    vmlal.s16       q2,  d12, d23       // a0 l, b0 l -> acc00
    vstr    d7, [sp, #8]
    vmov    q6,  q10
    vmov    d7,  d31    // a1 h
    vmov    d31, d24    // a4 l
    vmov    d24, d15    // b1 h
    vmov    d30, d25    // a4 h
    vld1.8  {d21}, [r3]
    vmov    d25, d14    // b1 l
    vmlal.s16       q6, d7,  d24        // a1 h, b1 h -> acc01
    vmlal.s16       q4, d29, d23        // a3 l, b0 l -> acc30
    vmlal.s16       q5, d28, d22        // a3 h, b0 h -> acc31
    vmlal.s16       q8, d27, d23        // a2 l, b0 l -> acc20
    vmlal.s16       q1, d26, d22        // a2 h, b0 h -> acc21
    vldr    d14, [sp, #8]
    vmlal.s16       q9, d6, d23         // a1 l, b0 l -> acc10
    vmlal.s16       q0, d7, d22         // a1 h, b0 h -> acc11
    vmovl.s8        q7, d14             // b2
    vmlal.s16       q4, d31, d25        // a4 l, b1 l -> acc30
    vmlal.s16       q5, d30, d24        // a4 h, b1 h -> acc31
    vmlal.s16       q8, d29, d25        // a3 l, b1 l -> acc20
    vmlal.s16       q1, d28, d24        // a3 h, b1 h -> acc21
    vmlal.s16       q9, d27, d25        // a2 l, b1 l -> acc10
    vmlal.s16       q0, d26, d24        // a2 h, b1 h -> acc11
    vmlal.s16       q2, d6, d25         // a1 l, b1 l -> acc00
    vmovl.s8        q12, d21            // a5
    vmov    q10, q6
    subs    r14, r14, #1
    add     src, src, src_y_step
    vmlal.s16       q8, d31, d14        // a4 l, b2 l -> acc20
    vmlal.s16       q1, d30, d15        // a4 h, b2 h -> acc21
    vmlal.s16       q9, d29, d14        // a3 l, b2 l -> acc10
    vmlal.s16       q0, d28, d15        // a3 h, b2 h -> acc11
    vmlal.s16       q2, d27, d14        // a2 l, b2 l -> acc00
    vmlal.s16       q10, d26, d15       // a2 h, b2 h -> acc01
    vmlal.s16       q4, d24, d14        // a5 l, b2 l -> acc30
    vmlal.s16       q5, d25, d15        // a4 h, b2 h -> acc31
    bne     .L2

add     r9, r10, #16
add     r10, scale_z, r10
add     scale_z, scale_z, r9
vld1.32 {d28-d29}, [scale_z]
vcvt.f32.s32    q10, q10
ldr     r3, .L6
vld1.32 {d30-d31}, [r10]
vdup.32 q3, r3
vmul.f32        q10, q10, q14
vcvt.f32.s32    q2, q2
vcvt.f32.s32    q0, q0
vcvt.f32.s32    q8, q8
vcvt.f32.s32    q1, q1
vmov q13, q3
vmul.f32        q8, q8, q15
vcvt.f32.s32    q9, q9
vmul.f32        q2, q2, q15
vcvt.f32.s32    q4, q4
vcvt.f32.s32    q5, q5
vadd.f32        q10, q10, q3
vmul.f32        q0, q0, q14
vmul.f32        q1, q1, q14
vmul.f32        q12, q9, q15
vadd.f32        q11, q8, q3
vadd.f32        q2, q2, q3
vmul.f32        q8, q5, q14
vmul.f32        q9, q4, q15
vsub.i32        q10, q10, q13
vadd.f32        q0, q0, q3
vadd.f32        q1, q1, q3
vsub.i32        q2, q2, q13
vadd.f32        q12, q12, q3
vadd.f32        q9, q9, q3
vqmovn.s32      d20, q10
vsub.i32        q0, q0, q13
vsub.i32        q14, q1, q13
vadd.f32        q8, q8, q3
vsub.i32        q12, q12, q13
vsub.i32        q11, q11, q13
vsub.i32        q8, q8, q13
vqmovn.s32      d4, q2
vsub.i32        q9, q9, q13
vmov    d5, d20
vqmovn.s32      d0, q0
vqmovn.s32      d28, q14
add     r0, r0, r11
mov     r3, r0
vqmovn.s32      d24, q12
vqmovn.s32      d22, q11
vqmovn.s32      d18, q9
vqmovn.s16      d4, q2
vmov    d25, d0
vmov    d23, d28
vqmovn.s32      d16, q8
vst1.8  {d4}, [r3], dst_depth
vmov    d17, d16
lsl     r5, dst_depth, #1
vqmovn.s16      d24, q12
vqmovn.s16      d22, q11
vmov    d16, d18
add     r5, r0, r5
vst1.8  {d24}, [r3]
vqmovn.s16      d16, q8
vst1.8  {d22}, [r5], dst_depth
vst1.8  {d16}, [r5]

add  sp, sp, #16
vpop {q4-q7}
pop  {r4, r5, r6, r7, r8, r9, r10, r11, pc}

.L6:
    .word 0x4B400000

#endif
#endif
