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

asm_function GEMM_FP16_N8
//void GEMM_FP16_N8(fp16_t* dst, const fp16_t* src, const fp16_t* weight, int src_depth,
//                            int dst_step, int dst_depth, int width, fp16_t *bias, int32_t relu)

.macro COMPUTE_UNIT_0 z0 z1 z2 z3 z4 z5 z6 z7 y
vmla.f16 \z0, \y, d0[0]
vmla.f16 \z1, \y, d0[1]
vmla.f16 \z2, \y, d0[2]
vmla.f16 \z3, \y, d0[3]
vmla.f16 \z4, \y, d1[0]
vmla.f16 \z5, \y, d1[1]
vmla.f16 \z6, \y, d1[2]
vmla.f16 \z7, \y, d1[3]
.endm

.macro COMPUTE_UNIT_1 z0 z1 z2 z3 z4 z5 z6 z7 y
vmla.f16 \z0, \y, d2[0]
vmla.f16 \z1, \y, d2[1]
vmla.f16 \z2, \y, d2[2]
vmla.f16 \z3, \y, d2[3]
vmla.f16 \z4, \y, d3[0]
vmla.f16 \z5, \y, d3[1]
vmla.f16 \z6, \y, d3[2]
vmla.f16 \z7, \y, d3[3]
.endm

.macro COMPUTE_UNIT_2 z0 z1 z2 z3 z4 z5 z6 z7 y
vmla.f16 \z0, \y, d4[0]
vmla.f16 \z1, \y, d4[1]
vmla.f16 \z2, \y, d4[2]
vmla.f16 \z3, \y, d4[3]
vmla.f16 \z4, \y, d5[0]
vmla.f16 \z5, \y, d5[1]
vmla.f16 \z6, \y, d5[2]
vmla.f16 \z7, \y, d5[3]
.endm

.macro COMPUTE_UNIT_3 z0 z1 z2 z3 z4 z5 z6 z7 y
vmla.f16 \z0, \y, d6[0]
vmla.f16 \z1, \y, d6[1]
vmla.f16 \z2, \y, d6[2]
vmla.f16 \z3, \y, d6[3]
vmla.f16 \z4, \y, d7[0]
vmla.f16 \z5, \y, d7[1]
vmla.f16 \z6, \y, d7[2]
vmla.f16 \z7, \y, d7[3]
.endm

.macro COMPUTE_M4_UNIT_0 z0 z1 z2 z3 y
vmla.f16 \z0, \y, d0[0]
vmla.f16 \z1, \y, d0[1]
vmla.f16 \z2, \y, d0[2]
vmla.f16 \z3, \y, d0[3]
.endm

.macro COMPUTE_M4_UNIT_1 z0 z1 z2 z3 y
vmla.f16 \z0, \y, d1[0]
vmla.f16 \z1, \y, d1[1]
vmla.f16 \z2, \y, d1[2]
vmla.f16 \z3, \y, d1[3]
.endm

.macro COMPUTE_M4_UNIT_2 z0 z1 z2 z3 y
vmla.f16 \z0, \y, d2[0]
vmla.f16 \z1, \y, d2[1]
vmla.f16 \z2, \y, d2[2]
vmla.f16 \z3, \y, d2[3]
.endm

.macro COMPUTE_M4_UNIT_3 z0 z1 z2 z3 y
vmla.f16 \z0, \y, d3[0]
vmla.f16 \z1, \y, d3[1]
vmla.f16 \z2, \y, d3[2]
vmla.f16 \z3, \y, d3[3]
.endm

.macro COMPUTE_M3_UNIT_0 z0 z1 z2 y
vmla.f16 \z0, \y, d0[0]
vmla.f16 \z1, \y, d0[1]
vmla.f16 \z2, \y, d0[2]
.endm

.macro COMPUTE_M3_UNIT_1 z0 z1 z2 y
vmla.f16 \z0, \y, d1[0]
vmla.f16 \z1, \y, d1[1]
vmla.f16 \z2, \y, d1[2]
.endm

.macro COMPUTE_M3_UNIT_2 z0 z1 z2 y
vmla.f16 \z0, \y, d2[0]
vmla.f16 \z1, \y, d2[1]
vmla.f16 \z2, \y, d2[2]
.endm

.macro COMPUTE_M3_UNIT_3 z0 z1 z2 y
vmla.f16 \z0, \y, d3[0]
vmla.f16 \z1, \y, d3[1]
vmla.f16 \z2, \y, d3[2]
.endm

push {r4-r11, lr}
vpush {q4-q7}

// Auto Load:
// r0:dst, r1:src, r2:weight, r3:src_depth

// Load from sp:
// r4:dst_step, r5:dst_depth, r6:width, r7:bias, r12:store_flag
ldr r4, [sp, #100]
ldr r5, [sp, #104]
ldr r6, [sp, #108]
ldr r7, [sp, #112]
ldr r12, [sp, #116]

//dst_step multi by sizeof(__fp16)
lsl r4, r4, #1

// r11:weight_step (src_depth * 8 * sizeof(__fp16))
lsl r11, r3, #4

// src_depth align with 8
lsr r8, r3, #3
lsl r8, r8, #3
// src_depth remain
sub r9, r3, r8
mov r3, r8
mov r10, r9

// store src_ptr to [sp, #-4]
str r1, [sp, #-4]

LoopDz:
// dst_ptr tmp, store to [sp, #-8]
str r0, [sp, #-8]
// weight ptr tmp, [sp, #-12]
str r2, [sp, #-12]

L8:
    cmp r6, #7
    ble L4

    vld1.16 q8, [r7]
    pld [r1, #512]
    vld1.16 q0, [r1]!
    pld [r2, #512]
    vld1.16 q4, [r2]!

    vmov q9,  q8
    vmov q10, q8
    vmov q11, q8
    vmov q12, q8
    vmov q13, q8
    vmov q14, q8
    vmov q15, q8

    cmp r8, #7
    ble L8MAC8CEND

L8MAC8C:
    vld1.16 {q1, q2}, [r1]!
    vld1.16 {q5, q6}, [r2]!
    // oc8ic0 * [ic0m0, ic0m7]
    COMPUTE_UNIT_0 q8, q9, q10, q11, q12, q13, q14, q15, q4

    vld1.16 q3, [r1]!
    vld1.16 q0, [r1]!
    vld1.16 q7, [r2]!
    vld1.16 q4, [r2]!
    // oc8ic1 * [ic1m0, ic1m7]
    COMPUTE_UNIT_1 q8, q9, q10, q11, q12, q13, q14, q15, q5

    vld1.16 q1, [r1]!
    vld1.16 q5, [r2]!
    // oc8ic2 * [ic2m0, ic2m7]
    COMPUTE_UNIT_2 q8, q9, q10, q11, q12, q13, q14, q15, q6

    vld1.16 q2, [r1]!
    vld1.16 q6, [r2]!
    // oc8ic3 * [ic3m0, ic3m7]
    COMPUTE_UNIT_3 q8, q9, q10, q11, q12, q13, q14, q15, q7

    vld1.16 q3, [r1]!
    vld1.16 q7, [r2]!
    // oc8ic4 * [ic4m0, ic4m7]
    COMPUTE_UNIT_0 q8, q9, q10, q11, q12, q13, q14, q15, q4

    pld [r1, #512]
    vld1.16 q0, [r1]!
    // oc8ic5 * [ic5m0, ic5m7]
    COMPUTE_UNIT_1 q8, q9, q10, q11, q12, q13, q14, q15, q5

    pld [r2, #512]
    vld1.16 q4, [r2]!
    // oc8ic6 * [ic6m0, ic6m7]
    COMPUTE_UNIT_2 q8, q9, q10, q11, q12, q13, q14, q15, q6

    subs r3, r3, #8
    // oc8ic7 * [ic7m0, ic7m7]
    COMPUTE_UNIT_3 q8, q9, q10, q11, q12, q13, q14, q15, q7

    bne L8MAC8C

L8MAC8CEND:
    cmp r10, #0
    // sub pre-load offset
    sub r1, r1, #16
    sub r2, r2, #16
    beq L8MAC1CEND

L8MAC1C:
    // add pre-load offset
    add r1, r1, #16
    add r2, r2, #16
    subs r9, #1
    COMPUTE_UNIT_0 q8, q9, q10, q11, q12, q13, q14, q15, q4
    vld1.16 {q0}, [r1]
    vld1.16 {q4}, [r2]
    bne L8MAC1C

L8MAC1CEND:
    // if relu
    cmp r12, #0
    beq Store8
    veor q0, q0, q0
    vmax.f16 q8,  q8,  q0
    vmax.f16 q9,  q9,  q0
    vmax.f16 q10, q10, q0
    vmax.f16 q11, q11, q0
    vmax.f16 q12, q12, q0
    vmax.f16 q13, q13, q0
    vmax.f16 q14, q14, q0
    vmax.f16 q15, q15, q0

    // if relu6
    cmp r12, #2
    bne Store8
    // 6.0f
    vmov.i16 q1, #0x4600
    vmin.f16 q8,  q8,  q1
    vmin.f16 q9,  q9,  q1
    vmin.f16 q10, q10, q1
    vmin.f16 q11, q11, q1
    vmin.f16 q12, q12, q1
    vmin.f16 q13, q13, q1
    vmin.f16 q14, q14, q1
    vmin.f16 q15, q15, q1
Store8:
    vstm r0!, {d16-d23}
    sub r6, r6, #8
    vstm r0!, {d24-d31}

    cmp r6, #8
    // reset weight ptr
    ldr r2, [sp, #-12]
    // reset loop counter
    mov r3, r8
    mov r9, r10

    bge L8

L4:
    cmp r6, #3
    ble L1

    vld1.16 q8, [r7]
    pld [r1, #256]
    vldr d0, [r1, #0]
    pld [r2, #512]
    vld1.16 q4, [r2]!
    vmov q9,  q8
    vmov q10, q8
    vmov q11, q8

    cmp r8, #7
    ble L4MAC8CEND

L4MAC8C:
    vldr d1, [r1, #8]
    vldr d2, [r1, #16]
    vld1.16 {q5, q6}, [r2]!
    // oc8ic0 * [ic0m0, ic0m3]
    COMPUTE_M4_UNIT_0 q8, q9, q10, q11, q4

    vldr d3, [r1, #24]
    vldr d0, [r1, #32]
    vld1.16 {q7}, [r2]!
    vld1.16 {q4}, [r2]!
    // oc8ic1 * [ic1m0, ic1m3]
    COMPUTE_M4_UNIT_1 q8, q9, q10, q11, q5

    vldr d1, [r1, #40]
    vld1.16 {q5}, [r2]!
    // oc8ic2 * [ic2m0, ic2m3]
    COMPUTE_M4_UNIT_2 q8, q9, q10, q11, q6

    vldr d2, [r1, #48]
    vld1.16 {q6}, [r2]!
    // oc8ic3 * [ic3m0, ic3m3]
    COMPUTE_M4_UNIT_3 q8, q9, q10, q11, q7

    vldr d3, [r1, #56]
    vld1.16 {q7}, [r2]!
    add r1, r1, #64
    // oc8ic4 * [ic4m0, ic4m3]
    COMPUTE_M4_UNIT_0 q8, q9, q10, q11, q4

    pld [r1, #256]
    vldr d0, [r1, #0]
    // oc8ic5 * [ic5m0, ic5m3]
    COMPUTE_M4_UNIT_1 q8, q9, q10, q11, q5

    pld [r2, #512]
    vld1.16 {q4}, [r2]!
    // oc8ic6 * [ic6m0, ic6m3]
    COMPUTE_M4_UNIT_2 q8, q9, q10, q11, q6

    subs r3, r3, #8
    // oc8ic7 * [ic7m0, ic7m3]
    COMPUTE_M4_UNIT_3 q8, q9, q10, q11, q7

    bne L4MAC8C

L4MAC8CEND:
    cmp r10, #0
    beq L4MAC1CEND

L4MAC1C:
    // add pre-load offset
    add r1, r1, #8
    subs r9, r9, #1
    // oc8ic0 * [ic0m0, ic0m3]
    COMPUTE_M4_UNIT_0 q8, q9, q10, q11, q4
    vldr d0, [r1]
    vld1.16 {q4}, [r2]!
    bne L4MAC1C

L4MAC1CEND:
    // if relu
    cmp r12, #0
    beq Store4
    veor q0, q0, q0
    vmax.f16 q8,  q8,  q0
    vmax.f16 q9,  q9,  q0
    vmax.f16 q10, q10, q0
    vmax.f16 q11, q11, q0

    // if relu6
    cmp r12, #2
    bne Store4
    // 6.0f
    vmov.i16 q1, #0x4600
    vmin.f16 q8,  q8,  q1
    vmin.f16 q9,  q9,  q1
    vmin.f16 q10, q10, q1
    vmin.f16 q11, q11, q1
Store4:
    vstm r0!, {d16-d23}
    sub r6, r6, #4

    cmp r6, #4
    // reset weight ptr
    ldr r2, [sp, #-12]
    // reset loop counter
    mov r3, r8
    mov r9, r10

    bge L4

L1:
    cmp r6, #0
    ble END

    vld1.16 q8, [r7]
    pld [r1, #256]
    // when L1, src is 4 x crr
    vldr d0, [r1, #0]
    pld [r2, #512]
    vld1.16 {q4}, [r2]!

    vmov q9,  q8
    vmov q10, q8

    cmp r8, #7
    ble L1MAC8CEND

L1MAC8C:
    vldr d1, [r1, #8]
    vldr d2, [r1, #16]
    vld1.16 {q5, q6}, [r2]!
    // oc8ic0 * [ic0m0, ic0m2]
    COMPUTE_M3_UNIT_0 q8, q9, q10, q4

    vldr d3, [r1, #24]
    vldr d0, [r1, #32]
    vld1.16 {q7}, [r2]!
    vld1.16 {q4}, [r2]!
    // oc8ic1 * [ic1m0, ic1m2]
    COMPUTE_M3_UNIT_1 q8, q9, q10, q5

    vldr d1, [r1, #40]
    vld1.16 {q5}, [r2]!
    // oc8ic2 * [ic2m0, ic2m2]
    COMPUTE_M3_UNIT_2 q8, q9, q10, q6

    vldr d2, [r1, #48]
    vld1.16 {q6}, [r2]!
    // oc8ic3 * [ic3m0, ic3m2]
    COMPUTE_M3_UNIT_3 q8, q9, q10, q7

    vldr d3, [r1, #56]
    vld1.16 {q7}, [r2]!
    add r1, r1, #64
    // oc8ic4 * [ic4m0, ic4m2]
    COMPUTE_M3_UNIT_0 q8, q9, q10, q4

    pld [r1, #256]
    vldr d0, [r1, #0]
    // oc8ic5 * [ic5m0, ic5m2]
    COMPUTE_M3_UNIT_1 q8, q9, q10, q5

    pld [r2, #512]
    vld1.16 {q4}, [r2]!
    // oc8ic6 * [ic6m0, ic6m2]
    COMPUTE_M3_UNIT_2 q8, q9, q10, q6

    subs r3, r3, #8
    // oc8ic7 * [ic7m0, ic7m2]
    COMPUTE_M3_UNIT_3 q8, q9, q10, q7

    bne L1MAC8C

L1MAC8CEND:
    cmp r10, #0
    beq L1MAC1CEND

L1MAC1C:
    // add pre-load offset
    add r1, r1, #8
    subs r9, r9, #1
    // oc8ic0 * [ic0m0, ic0m2]
    COMPUTE_M3_UNIT_0 q8, q9, q10, q4
    vldr d0, [r1]
    vld1.16 {q4}, [r2]!
    bne L1MAC1C

L1MAC1CEND:
    // if relu
    cmp r12, #0
    beq Store1
    veor q0, q0, q0
    vmax.f16 q8,  q8,  q0
    vmax.f16 q9,  q9,  q0
    vmax.f16 q10, q10, q0

    // if relu6
    cmp r12, #2
    bne Store1
    // 6.0f
    vmov.i16 q1, #0x4600
    vmin.f16 q8,  q8,  q1
    vmin.f16 q9,  q9,  q1
    vmin.f16 q10, q10, q1
Store1:
    cmp r6, #3
    beq Store1_3reg
    cmp r6, #2
    beq Store1_2reg
    cmp r6, #1
    beq Store1_1reg
Store1_3reg:
    vstm r0!, {d16-d21}
    b Store1_End
Store1_2reg:
    vstm r0!, {d16-d19}
    b Store1_End
Store1_1reg:
    vstm r0!, {d16-d17}
Store1_End:
    // reset weight ptr
    ldr r2, [sp, #-12]
    // reset loop counter
    mov r3, r8
    mov r9, r10

END:

subs r5, r5, #8
// update bias_ptr
add r7, r7, #16
// update dst_ptr
ldr r0, [sp, #-8]
add r0, r0, r4
// reset src ptr
ldr r1, [sp, #-4]
// update weight ptr
ldr r2, [sp, #-12]
add r2, r2, r11
// reset M counter
ldr r6, [sp, #108]
bne LoopDz

vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif
#endif
