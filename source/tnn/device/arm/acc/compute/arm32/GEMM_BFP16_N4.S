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

asm_function GEMM_BFP16_N4
//void GEMM_BFP16_N4(bfp16_t* dst,               //r0: dst 
//                   const bfp16_t* src,         //r1: src
//                   const float* weight,      //r2: weight
//                   int src_depth_quad,       //r3: src_depth_quad
//                   int dst_step              //r4: dst_step            load from stack
//                   int dst_depth_quad,       //r5: dst_depth_quad      load from stack
//                   int width,                //r6: width               load from stack
//                   float *bias               //r7: bias
//                   int relu);                //reuse r12: relu?

.macro COMPUTE_UNIT_0 z0 z1 z2 z3 y
vmla.f32 \z0, \y, d0[0]
vmla.f32 \z1, \y, d0[1]
vmla.f32 \z2, \y, d1[0]
vmla.f32 \z3, \y, d1[1]
.endm

.macro COMPUTE_UNIT_1 z0 z1 z2 z3 y
vmla.f32 \z0, \y, d2[0]
vmla.f32 \z1, \y, d2[1]
vmla.f32 \z2, \y, d3[0]
vmla.f32 \z3, \y, d3[1]
.endm

.macro COMPUTE_UNIT_2 z0 z1 z2 z3 y
vmla.f32 \z0, \y, d4[0]
vmla.f32 \z1, \y, d4[1]
vmla.f32 \z2, \y, d5[0]
vmla.f32 \z3, \y, d5[1]
.endm

.macro COMPUTE_UNIT_3 z0 z1 z2 z3 y
vmla.f32 \z0, \y, d6[0]
vmla.f32 \z1, \y, d6[1]
vmla.f32 \z2, \y, d7[0]
vmla.f32 \z3, \y, d7[1]
.endm

dst            .req r0
src            .req r1
weight         .req r2
src_depth_quad .req r3
dst_step       .req r4
dst_depth_quad .req r5
width          .req r6
bias           .req r7

push {r4-r11, lr}
vpush {q4-q7}

//Auto Load:
//r0:dst, r1:src, r2:weight, r3:src_depth_quad

//Load from sp
//r4:dst_step, r5:dst_depth_quad, r6:width, r7:bias
ldr dst_step, [sp, #100]
ldr r5, [sp, #104]
ldr width, [sp, #108]
ldr bias, [sp, #112]


//step multi by sizeof(bfp16)
mov r12, #2
mul dst_step, r12, dst_step

//src_z_step
mov r12, #8
mul r8, r12, width

//weight_z_step
mov r12, #64
mul r9, r12, r3

//save outside loop, src_depth_quad, width
vmov.i32 d14[0], src_depth_quad
vmov.i32 d14[1], width

LoopDz:
vmov.i32 d15[1], dst
mov r10, src
mov r11, weight

L8:
cmp width, #7
ble L4

mov r12, src
vld1.32 {q8}, [bias]
vmov q9,  q8
vmov q10, q8
vld1.32 {q4}, [weight]!
vmov q11, q8
vmov q12, q8
vmov q13, q8
vld1.32 {q0}, [src]!
vshll.u16  q1, d1, #16
vshll.u16  q0, d0, #16
vmov q14, q8
vmov q15, q8

vmov.i32 d15[0], weight

L8Loop:
    pld [weight, #256]
    COMPUTE_UNIT_0 q8, q9, q10, q11, q4
    pld [src, #256]
    vld1.32 {q5}, [weight]!
    vld1.32 {q2}, [src]!
    vshll.u16  q3, d5, #16
    vshll.u16  q2, d4, #16
    COMPUTE_UNIT_1 q12, q13, q14, q15, q4

    vld1.32 {q4}, [weight]!
    COMPUTE_UNIT_2 q8, q9, q10, q11, q5
    vld1.32 {q0}, [src]!
    vshll.u16  q1, d1, #16
    vshll.u16  q0, d0, #16
    pld [weight, #256]
    COMPUTE_UNIT_3 q12, q13, q14, q15, q5

    vld1.32 {q5}, [weight]!
    COMPUTE_UNIT_0 q8, q9, q10, q11, q4
    vld1.32 {q2}, [src]!
    vshll.u16  q3, d5, #16
    vshll.u16  q2, d4, #16
    COMPUTE_UNIT_1 q12, q13, q14, q15, q4

    subs r3, r3, #1

    pld [src, #256]
    vld1.32 {q4}, [weight]!
    COMPUTE_UNIT_2 q8, q9, q10, q11, q5
    sub src, src, #64
    add src, src, r8

    vld1.32 {q0}, [src]!
    vshll.u16  q1, d1, #16
    vshll.u16  q0, d0, #16
    COMPUTE_UNIT_3 q12, q13, q14, q15, q5
    bne L8Loop

add src, r12, #64
ldr r12, [sp, #116]
sub width, width, #8
vmov.i32 weight, d15[0]
vmov.i32 r3, d14[0]

cmp r12, #1
blt Store8
vmov.i32 q0, #0
vmax.f32 q8, q8, q0
vmax.f32 q9, q9, q0
vmax.f32 q10, q10, q0
vmax.f32 q11, q11, q0
vmax.f32 q12, q12, q0
vmax.f32 q13, q13, q0
vmax.f32 q14, q14, q0
vmax.f32 q15, q15, q0
Store8:
cmp width, #8
vshrn.u32 d16, q8, #16
vshrn.u32 d17, q9, #16
vshrn.u32 d18, q10, #16
vshrn.u32 d19, q11, #16
vstm dst!, {d16-d19}
vshrn.u32 d24, q12, #16
vshrn.u32 d25, q13, #16
vshrn.u32 d26, q14, #16
vshrn.u32 d27, q15, #16
vstm dst!, {d24-d27}

bge L8

L4:
cmp width, #3
ble L1

mov r12, src
vld1.32 {q8}, [bias]
vld1.32 {q0}, [src]!
vld1.32 {q4, q5}, [weight]!
vshll.u16  q1, d1, #16
vshll.u16  q0, d0, #16
vmov q9, q8
vmov q10, q8
vmov q11, q8

vmov.i32 d15[0], weight
L4Loop:
    pld [src, #256]
    COMPUTE_UNIT_0 q8, q9, q10, q11, q4
    vld1.32 {q2}, [src]!
    vshll.u16  q3, d5, #16
    vshll.u16  q2, d4, #16
    pld [weight, #256]
    COMPUTE_UNIT_1 q8, q9, q10, q11, q5
    vld1.32 {q12, q13}, [weight]!
    COMPUTE_UNIT_2 q8, q9, q10, q11, q12
    pld [weight, #256]
    vld1.32 {q4, q5}, [weight]!
    pld [src, #256]
    COMPUTE_UNIT_3 q8, q9, q10, q11, q13

    subs r3, r3, #1

    sub src, src, #32
    add src, src, r8
    pld [src, #256]
    vld1.32 {q0}, [src]!
    vshll.u16  q1, d1, #16
    vshll.u16  q0, d0, #16
    pld [weight, #256]
    bne L4Loop

add src, r12, #32
ldr r12, [sp, #116]
subs width, width, #4
vmov.i32 weight, d15[0]
vmov.i32 r3, d14[0]

cmp r12, #1
blt Store4
vmov.i32 q0, #0
vmax.f32 q8, q8, q0
vmax.f32 q9, q9, q0
vmax.f32 q10, q10, q0
vmax.f32 q11, q11, q0

Store4:
cmp width, #4
vshrn.u32 d16, q8, #16
vshrn.u32 d17, q9, #16
vshrn.u32 d18, q10, #16
vshrn.u32 d19, q11, #16
vstm dst!, {d16-d19}
bge L4

L1:
cmp width, #0
ble End

L1Loop:
    mov r12, src
    vld1.32 {q9}, [bias] 
    vmov.i32 d15[0], weight
    pld [src, #256]
    vld1.32 {d0}, [src], r8
    vshll.u16  q0, d0, #16
    pld [weight, #256]
    vldm weight!, {d6-d13}
    vmul.f32 q8, q3, d0[0]
    subs r3, r3, #1
    vmla.f32 q9, q4, d0[1]

    beq L1LoopZEnd

    L1LoopZ:
        vmla.f32 q8, q5, d1[0]
        vmla.f32 q9, q6, d1[1]
        pld [src, #256]
        vld1.32 {d0}, [src], r8
        vshll.u16  q0, d0, #16
        pld [weight, #256]
        vldm weight!, {d6-d13}

        subs r3, r3, #1
        vmla.f32 q8, q3, d0[0]
        vmla.f32 q9, q4, d0[1]

        bne L1LoopZ
    L1LoopZEnd:

add src, r12, #8
ldr r12, [sp, #116]
vmla.f32 q8, q5, d1[0]
vmla.f32 q9, q6, d1[1]
vadd.f32 q8, q8, q9

vmov.i32 weight, d15[0]
vmov.i32 r3, d14[0]

cmp r12, #1
blt Store1
vmov.i32 q0, #0
vmax.f32 q8, q8, q0
Store1:
    subs width, width, #1
    vshrn.u32 d16, q8, #16
    vst1.f32 {d16}, [dst]!
bne L1Loop

End:

mov src, r10
subs r5, r5, #1
vmov.i32 r10, d15[1]
add dst, r10, dst_step
add weight, r11, r9
add bias, bias, #16
vmov.i32 width, d14[1]
bne LoopDz

vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif
