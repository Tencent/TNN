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

.macro TRANSPOSE_2X16_S8 r0 r1
// r0: w0c0,w0c1,w0c2,w0c3,w0c4,w0c5,w0c6,w0c7 | w2c0,w2c1,w2c2,w2c3,w2c4,w2c5,w2c6,w2c7
// r1: w1c0,w1c1,w1c2,w1c3,w1c4,w1c5,w1c6,w1c7 | 0
    // w0c0,w1c0,w0c1,w1c1,w0c2,w1c2,w0c3,w1c3 | w0c4,w1c4,w0c5,w1c5,w0c6,w1c6,w0c7,w1c7
    // w2c0,0,   w2c1,0,   w2c2,0,   w2c3,0,   | w2c4,0,   w2c5,0,   w2c6,0,   w2c7,0
    vzip.8 \r0, \r1
    // w0c0,w1c0,w2c0,0,w0c1,w1c1,w2c1,0,w0c2,w1c2,w2c2,0,w0c3,w1c3,w2c3,0
    // w0c4,w1c4,w2c4,0,w0c5,w1c5,w2c5,0,w0c6,w1c6,w2c6,0,w0c7,w1c7,w2c7,0
    vzip.16 \r0, \r1
.endm

.macro TRANSPOSE_2X8_S8 r0 r1
// r0: w0c0,w0c1,w0c2,w0c3 | w2c0,w2c1,w2c2,w2c3
// r1: w1c0,w1c1,w1c2,w1c3 | 0
    // w0c0,w1c0,w0c1,w1c1,w0c2,w1c2,w0c3,w1c3
    // w2c0,0,   w2c1,0,   w2c2,0,   w2c3,0
    vzip.8 \r0, \r1
    // w0c0,w1c0,w2c0,0,w0c1,w1c1,w2c1,0,w0c2,w1c2,w2c2,0,w0c3,w1c3,w2c3,0
    vzip.16 \r0, \r1
.endm

asm_function ConvDw3x3Int8SdotSlideW
//void ConvDw3x3Int8SdotSlideW(int8_t *dst_z,
//                        int8_t **src,
//                        const int8_t* weight_z,
//                        const int32_t* bias_z,
//                        const float* scale_z,
//                        long dc,
//                        long dst_depth,
//                        long width)
//r0(dst_z),
//r1(int8_t** src),
//r2(weight_z),
//r3(bias_z)

push {r4-r11, lr}
vpush {q4-q7}
// sp offset 9 x 4 + 16 x 4 = 100

//from stack(scale_z)   [sp, #100]
//from stack(dc)        [sp, #104]
//from stack(dst_depth) [sp, #108]
//from stack(width)     [sp, #112]
ldr r4, [sp, #100]
ldr r5, [sp, #104]
ldr r6, [sp, #108]
ldr r7, [sp, #112]

cmp r7, #0
ble End

// weight q0 - q5
// c0 k0k1k2-, c1 k0k1k2-, c2 k0k1k2-, c3 k0k1k2-
// c4 k0k1k2-, c5 k0k1k2-, c6 k0k1k2-, c7 k0k1k2-
vldm r2, {d0-d11}

// q6, q7 for convert fp32 to int32
vmov.f32 q6, #0.5
vmov.f32 q7, #-0.5

ldr r9,  [r1]
ldr r10, [r1, #4]
ldr r11, [r1, #8]
add r9, r5     // h0 ptr += dc
add r10, r5    // h1 ptr += dc
add r11, r5    // h2 ptr += dc

veor d19, d19, d19
vld1.8 {d16}, [r9], r6
vld1.8 {d18}, [r10], r6
vld1.8 {d17}, [r11], r6
vld1.32 {q10, q11}, [r3]   // init from bias
TRANSPOSE_2X16_S8 q8, q9
.word 0xfc604dc0 // vsdot.s8 q10, q8, q0
.word 0xfc626dc2 // vsdot.s8 q11, q9, q1

veor d19, d19, d19
vld1.8 {d16}, [r9], r6
vld1.8 {d18}, [r10], r6
vld1.8 {d17}, [r11], r6
vld1.32 {q12, q13}, [r3]   // init from bias
TRANSPOSE_2X16_S8 q8, q9
.word 0xfc604dc4 // vsdot.s8 q10, q8, q2
.word 0xfc626dc6 // vsdot.s8 q11, q9, q3
.word 0xfc608dc0 // vsdot.s8 q12, q8, q0
.word 0xfc62adc2 // vsdot.s8 q13, q9, q1

LoopDw:
    veor d19, d19, d19
    vld1.8 {d16}, [r9], r6
    vld1.8 {d18}, [r10], r6
    vld1.8 {d17}, [r11], r6
    vld1.32 {q14, q15}, [r3]  // init from bias
    pld [r9]
    pld [r10]
    pld [r11]
    TRANSPOSE_2X16_S8 q8, q9
    .word 0xfc604dc8 // vsdot.s8 q10, q8, q4
    .word 0xfc626dca // vsdot.s8 q11, q9, q5
    .word 0xfc608dc4 // vsdot.s8 q12, q8, q2
    .word 0xfc62adc6 // vsdot.s8 q13, q9, q3
    .word 0xfc60cdc0 // vsdot.s8 q14, q8, q0
    .word 0xfc62edc2 // vsdot.s8 q15, q9, q1

    subs r7, r7, #1

    vld1.32 {q8, q9}, [r4] // load scale
    vcvt.f32.s32 q10, q10
    vcvt.f32.s32 q11, q11
    vmul.f32 q10, q10, q8  // result *= scale
    vmul.f32 q11, q11, q9

    vcge.f32 q8, q10, #0
    vcge.f32 q9, q11, #0
    vbsl.f32 q8, q6, q7
    vbsl.f32 q9, q6, q7

    vadd.f32 q10, q10, q8
    vadd.f32 q11, q11, q9
    vcvt.s32.f32 q10, q10
    vcvt.s32.f32 q11, q11
    vqmovn.s32 d20, q10
    vqmovn.s32 d21, q11
    vqmovn.s16 d20, q10
    vst1.8 {d20}, [r0], r6

    vmov q10, q12
    vmov q11, q13
    vmov q12, q14
    vmov q13, q15

    bne LoopDw

End:

vpop {q4-q7}
pop {r4-r11, pc}

asm_function ConvDw3x3Int8SdotSlideWLeftC4
//void ConvDw3x3Int8SdotSlideWLeftC4(int8_t *dst_z,
//                        int8_t **src,
//                        const int8_t* weight_z,
//                        const int32_t* bias_z,
//                        const float* scale_z,
//                        long dc,
//                        long dst_depth,
//                        long width)
//r0(dst_z),
//r1(int8_t** src),
//r2(weight_z),
//r3(bias_z)

push {r4-r11, lr}
vpush {q4-q7}
// sp offset 9 x 4 + 16 x 4 = 100

//from stack(scale_z)   [sp, #100]
//from stack(dc)        [sp, #104]
//from stack(dst_depth) [sp, #108]
//from stack(width)     [sp, #112]
ldr r4, [sp, #100]
ldr r5, [sp, #104]
ldr r6, [sp, #108]
ldr r7, [sp, #112]

cmp r7, #0
ble C4End

// weight q0 - q2
// c0 k0k1k2-, c1 k0k1k2-, c2 k0k1k2-, c3 k0k1k2-
vldm r2, {d0-d5}

// q6, q7 for convert fp32 to int32
vmov.f32 q6, #0.5
vmov.f32 q7, #-0.5

// bias
vld1.32 {q3}, [r3]

// scale
vld1.32 {q4}, [r4]

ldr r9,  [r1]
ldr r10, [r1, #4]
ldr r11, [r1, #8]
add r9, r5     // h0 ptr += dc
add r10, r5    // h1 ptr += dc
add r11, r5    // h2 ptr += dc

veor d17, d17, d17
vld1.32 {d16[0]}, [r9], r6
vld1.32 {d17[0]}, [r10], r6
vld1.32 {d16[1]}, [r11], r6
vmov q10, q3
TRANSPOSE_2X8_S8 d16, d17
.word 0xfc604dc0 // vsdot.s8 q10, q8, q0

veor d17, d17, d17
vld1.32 {d16[0]}, [r9], r6
vld1.32 {d17[0]}, [r10], r6
vld1.32 {d16[1]}, [r11], r6
vmov q11, q3
TRANSPOSE_2X8_S8 d16, d17
.word 0xfc604dc2 // vsdot.s8 q10, q8, q1
.word 0xfc606dc0 // vsdot.s8 q11, q8, q0

C4LoopDw:
    veor d17, d17, d17
    vld1.32 {d16[0]}, [r9], r6
    vld1.32 {d17[0]}, [r10], r6
    vld1.32 {d16[1]}, [r11], r6
    vmov q12, q3
    pld [r9]
    pld [r10]
    pld [r11]
    TRANSPOSE_2X8_S8 d16, d17
    .word 0xfc604dc4 // vsdot.s8 q10, q8, q2
    .word 0xfc606dc2 // vsdot.s8 q11, q8, q1
    .word 0xfc608dc0 // vsdot.s8 q12, q8, q0

    subs r7, r7, #1

    vcvt.f32.s32 q13, q10
    vmov q10, q11
    vmov q11, q12
    vmul.f32 q13, q13, q4  // result *= scale
    vcge.f32 q8, q13, #0
    vbsl.f32 q8, q6, q7
    vadd.f32 q13, q13, q8
    vcvt.s32.f32 q13, q13
    vqmovn.s32 d26, q13
    vqmovn.s16 d26, q13
    vst1.32 {d26[0]}, [r0], r6

    bne C4LoopDw

C4End:

vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif
#endif
