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

asm_function GemmInt8SdotUnit4x8Kernel
//void GemmInt8SdotUnit4x8Kernel(int8_t* dst, const int8_t* src, const int8_t* weight,
//                         long src_depth, long dst_depth, long hw, 
//                         const int32_t* bias, const float* scale,
//                         long relu, const int8_t* add_input, 
//                         const float* add_scale, const int8_t* relu6_max)
//r0(dst),
//r1(src),
//r2(weight),
//r3(src_depth)

push {r4-r11, lr}
vpush {q4-q7}
// sp offset 9 x 4 + 16 x 4 = 100

//from stack(dst_depth) [sp, #100]
//from stack(hw)        [sp, #104]
//from stack(bias)      [sp, #108]
//from stack(scale)     [sp, #112]
//from stack(relu)      [sp, #116]
//from stack(add_input) [sp, #120]
//from stack(add_scale) [sp, #124]
//from stack(relu6_max) [sp, #128]

ldr r4, [sp, #100]
ldr r5, [sp, #104]
ldr r6, [sp, #108]

LoopHW4:
    // if hw counter <= 3, skip
    cmp r5, #3
    ble LoopHW1

    // src_ptr 0 ~ 3
    mov r9,  r1
    add r10, r1, r3
    add r11, r1, r3, lsl#1
    add r12, r10, r3, lsl#1

    // load bias 32bytes, accumulator 8 (hw4 oc8) reg
    vld1.32 {q8, q9}, [r6]
    vmov q10, q8
    vmov q11, q9
    vmov q12, q8
    vmov q13, q9
    vmov q14, q8
    vmov q15, q9

    // src_depth counter
    mov r7, r3

    // weight_ptr
    mov r8, r2

    cmp r7, #15
    ble LoopCrr8

    vld1.8 {q0, q1}, [r8]!

    vld1.8 {q4}, [r9]!
    vld1.8 {q5}, [r10]!
    vld1.8 {q6}, [r11]!
    vld1.8 {q7}, [r12]!

    vld1.8 {q2, q3}, [r8]!

    .word 0xfe600d48 // vsdot.s8 q8,  q0, d8[0]
    .word 0xfe622d48 // vsdot.s8 q9,  q1, d8[0]
    .word 0xfe604d4a // vsdot.s8 q10, q0, d10[0]
    .word 0xfe626d4a // vsdot.s8 q11, q1, d10[0]
    .word 0xfe608d4c // vsdot.s8 q12, q0, d12[0]
    .word 0xfe62ad4c // vsdot.s8 q13, q1, d12[0]
    .word 0xfe60cd4e // vsdot.s8 q14, q0, d14[0]
    .word 0xfe62ed4e // vsdot.s8 q15, q1, d14[0]

    sub r7, #16

    LoopCrr16:
        cmp r7, #15
        ble LoopCrr16End

        vld1.8 {q0, q1}, [r8]!

        .word 0xfe640d68 // vsdot.s8 q8,  q2, d8[1]
        .word 0xfe662d68 // vsdot.s8 q9,  q3, d8[1]
        pld [r9, #64]
        .word 0xfe644d6a // vsdot.s8 q10, q2, d10[1]
        .word 0xfe666d6a // vsdot.s8 q11, q3, d10[1]
        pld [r10, #64]
        .word 0xfe648d6c // vsdot.s8 q12, q2, d12[1]
        .word 0xfe66ad6c // vsdot.s8 q13, q3, d12[1]
        pld [r11, #64]
        .word 0xfe64cd6e // vsdot.s8 q14, q2, d14[1]
        .word 0xfe66ed6e // vsdot.s8 q15, q3, d14[1]
        pld [r12, #64]

        vld1.8 {q2, q3}, [r8]!

        .word 0xfe600d49 // vsdot.s8 q8,  q0, d9[0]
        .word 0xfe622d49 // vsdot.s8 q9,  q1, d9[0]
        .word 0xfe604d4b // vsdot.s8 q10, q0, d11[0]
        .word 0xfe626d4b // vsdot.s8 q11, q1, d11[0]
        .word 0xfe608d4d // vsdot.s8 q12, q0, d13[0]
        .word 0xfe62ad4d // vsdot.s8 q13, q1, d13[0]
        .word 0xfe60cd4f // vsdot.s8 q14, q0, d15[0]
        .word 0xfe62ed4f // vsdot.s8 q15, q1, d15[0]

        vld1.8 {q0, q1}, [r8]!

        .word 0xfe640d69 // vsdot.s8 q8,  q2, d9[1]
        .word 0xfe662d69 // vsdot.s8 q9,  q3, d9[1]
        vld1.8 {q4}, [r9]!
        .word 0xfe644d6b // vsdot.s8 q10, q2, d11[1]
        .word 0xfe666d6b // vsdot.s8 q11, q3, d11[1]
        vld1.8 {q5}, [r10]!
        .word 0xfe648d6d // vsdot.s8 q12, q2, d13[1]
        .word 0xfe66ad6d // vsdot.s8 q13, q3, d13[1]
        vld1.8 {q6}, [r11]!
        .word 0xfe64cd6f // vsdot.s8 q14, q2, d15[1]
        .word 0xfe66ed6f // vsdot.s8 q15, q3, d15[1]
        vld1.8 {q7}, [r12]!

        vld1.8 {q2, q3}, [r8]!
        sub r7, r7, #16

        .word 0xfe600d48 // vsdot.s8 q8,  q0, d8[0]
        .word 0xfe622d48 // vsdot.s8 q9,  q1, d8[0]
        .word 0xfe604d4a // vsdot.s8 q10, q0, d10[0]
        .word 0xfe626d4a // vsdot.s8 q11, q1, d10[0]
        .word 0xfe608d4c // vsdot.s8 q12, q0, d12[0]
        .word 0xfe62ad4c // vsdot.s8 q13, q1, d12[0]
        .word 0xfe60cd4e // vsdot.s8 q14, q0, d14[0]
        .word 0xfe62ed4e // vsdot.s8 q15, q1, d14[0]

        b LoopCrr16

    LoopCrr16End:
        vld1.8 {q0, q1}, [r8]!
        .word 0xfe640d68 // vsdot.s8 q8,  q2, d8[1]
        .word 0xfe662d68 // vsdot.s8 q9,  q3, d8[1]
        .word 0xfe644d6a // vsdot.s8 q10, q2, d10[1]
        .word 0xfe666d6a // vsdot.s8 q11, q3, d10[1]
        .word 0xfe648d6c // vsdot.s8 q12, q2, d12[1]
        .word 0xfe66ad6c // vsdot.s8 q13, q3, d12[1]
        .word 0xfe64cd6e // vsdot.s8 q14, q2, d14[1]
        .word 0xfe66ed6e // vsdot.s8 q15, q3, d14[1]
        vld1.8 {q2, q3}, [r8]!
        .word 0xfe600d49 // vsdot.s8 q8,  q0, d9[0]
        .word 0xfe622d49 // vsdot.s8 q9,  q1, d9[0]
        .word 0xfe604d4b // vsdot.s8 q10, q0, d11[0]
        .word 0xfe626d4b // vsdot.s8 q11, q1, d11[0]
        .word 0xfe608d4d // vsdot.s8 q12, q0, d13[0]
        .word 0xfe62ad4d // vsdot.s8 q13, q1, d13[0]
        .word 0xfe60cd4f // vsdot.s8 q14, q0, d15[0]
        .word 0xfe62ed4f // vsdot.s8 q15, q1, d15[0]
        .word 0xfe640d69 // vsdot.s8 q8,  q2, d9[1]
        .word 0xfe662d69 // vsdot.s8 q9,  q3, d9[1]
        .word 0xfe644d6b // vsdot.s8 q10, q2, d11[1]
        .word 0xfe666d6b // vsdot.s8 q11, q3, d11[1]
        .word 0xfe648d6d // vsdot.s8 q12, q2, d13[1]
        .word 0xfe66ad6d // vsdot.s8 q13, q3, d13[1]
        .word 0xfe64cd6f // vsdot.s8 q14, q2, d15[1]
        .word 0xfe66ed6f // vsdot.s8 q15, q3, d15[1]
    
    LoopCrr8:
        cmp r7, #7
        ble LoopCrr4

        vld1.8 {d8}, [r9]!
        vld1.8 {d9}, [r10]!
        vld1.8 {d10}, [r11]!
        vld1.8 {d11}, [r12]!

        vld1.8 {q0, q1}, [r8]!
        vld1.8 {q2, q3}, [r8]!

        .word 0xfe600d48 // vsdot.s8 q8,  q0, d8[0]
        .word 0xfe622d48 // vsdot.s8 q9,  q1, d8[0]
        .word 0xfe604d49 // vsdot.s8 q10, q0, d9[0]
        .word 0xfe626d49 // vsdot.s8 q11, q1, d9[0]
        .word 0xfe608d4a // vsdot.s8 q12, q0, d10[0]
        .word 0xfe62ad4a // vsdot.s8 q13, q1, d10[0]
        .word 0xfe60cd4b // vsdot.s8 q14, q0, d11[0]
        .word 0xfe62ed4b // vsdot.s8 q15, q1, d11[0]

        sub r7, #8

        .word 0xfe640d68 // vsdot.s8 q8,  q2, d8[1]
        .word 0xfe662d68 // vsdot.s8 q9,  q3, d8[1]
        .word 0xfe644d69 // vsdot.s8 q10, q2, d9[1]
        .word 0xfe666d69 // vsdot.s8 q11, q3, d9[1]
        .word 0xfe648d6a // vsdot.s8 q12, q2, d10[1]
        .word 0xfe66ad6a // vsdot.s8 q13, q3, d10[1]
        .word 0xfe64cd6b // vsdot.s8 q14, q2, d11[1]
        .word 0xfe66ed6b // vsdot.s8 q15, q3, d11[1]

        b LoopCrr8
    
    LoopCrr4:
        cmp r7, #3
        ble LoopEnd

        vld1.32 {d8[0]}, [r9]!
        vld1.32 {d9[0]}, [r10]!
        vld1.32 {d10[0]}, [r11]!
        vld1.32 {d11[0]}, [r12]!
        vld1.8 {q0, q1}, [r8]!

        sub r7, #4
        .word 0xfe600d48 // vsdot.s8 q8,  q0, d8[0]
        .word 0xfe622d48 // vsdot.s8 q9,  q1, d8[0]
        .word 0xfe604d49 // vsdot.s8 q10, q0, d9[0]
        .word 0xfe626d49 // vsdot.s8 q11, q1, d9[0]
        .word 0xfe608d4a // vsdot.s8 q12, q0, d10[0]
        .word 0xfe62ad4a // vsdot.s8 q13, q1, d10[0]
        .word 0xfe60cd4b // vsdot.s8 q14, q0, d11[0]
        .word 0xfe62ed4b // vsdot.s8 q15, q1, d11[0]

        b LoopCrr4

LoopEnd:
    // hw counter -= 4
    sub r5, #4
    // src_ptr += 4 * src_depth
    add r1, r1, r3, lsl#2

    ldr r8, [sp, #112]  // scale_ptr
    ldr r7, [sp, #116]  // relu
    
    // scale oc0 ~ oc7
    vld1.32 {q0, q1}, [r8]

    ldr r8, [sp, #120]  // add_input

ConvReluAdd:
    cmp r7, #-1   // if relu == -1, Conv-Relu-Add
    bne MulScale

    veor q2, q2, q2
    vmax.s32 q8,  q8,  q2
    vmax.s32 q9,  q9,  q2
    vmax.s32 q10, q10, q2
    vmax.s32 q11, q11, q2
    vmax.s32 q12, q12, q2
    vmax.s32 q13, q13, q2
    vmax.s32 q14, q14, q2
    vmax.s32 q15, q15, q2
MulScale:
    vcvt.f32.s32 q8,  q8
    vcvt.f32.s32 q9,  q9
    vcvt.f32.s32 q10, q10
    vcvt.f32.s32 q11, q11
    vcvt.f32.s32 q12, q12
    vcvt.f32.s32 q13, q13
    vcvt.f32.s32 q14, q14
    vcvt.f32.s32 q15, q15

    vmul.f32 q8,  q8,  q0
    vmul.f32 q9,  q9,  q1
    vmul.f32 q10, q10, q0
    vmul.f32 q11, q11, q1
    vmul.f32 q12, q12, q0
    vmul.f32 q13, q13, q1
    vmul.f32 q14, q14, q0
    vmul.f32 q15, q15, q1

    cmp r8, #0       // if add_input == 0, skip
    beq ConvAddPost

AddInputScale:
    ldr r12, [sp, #124]   // add_scale
    // add_input_ptr 0 ~ 3
    add r9, r8, r4
    add r10, r8, r4, lsl#1
    add r11, r9, r4, lsl#1

    vldr d0, [r8]
    vldr d2, [r9]
    vldr d4, [r10]
    vldr d6, [r11]

    // add_input_ptr += 4 * dst_depth
    add r8, r8, r4, lsl#2

    // add_scale
    vld1.32 {q6, q7}, [r12]

    // convert add_input int8 to fp32
    vmovl.s8 q0, d0
    vmovl.s8 q1, d2
    vmovl.s8 q2, d4
    vmovl.s8 q3, d6
    vmovl.s16 q4, d0
    vmovl.s16 q5, d1
    vmovl.s16 q0, d2
    vmovl.s16 q1, d3
    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5
    vcvt.f32.s32 q0, q0
    vcvt.f32.s32 q1, q1

    vmla.f32 q8,  q4, q6   // result += add_input * add_scale
    vmovl.s16 q4, d4
    vmla.f32 q9,  q5, q7
    vmovl.s16 q5, d5
    vmla.f32 q10, q0, q6
    vmovl.s16 q0, d6
    vmla.f32 q11, q1, q7
    vmovl.s16 q1, d7

    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5
    vcvt.f32.s32 q0, q0
    vcvt.f32.s32 q1, q1

    vmla.f32 q12, q4, q6
    vmla.f32 q13, q5, q7
    vmla.f32 q14, q0, q6
    vmla.f32 q15, q1, q7

    str r8, [sp, #120]  // update add_input

ConvAddPost:
    // f32 --> s32 --> s8
    // val + (val >= 0.f ? 0.5f : -0.5f)
    vmov.f32 q0, #0.5
    vmov.f32 q1, #-0.5

    vcge.f32 q2, q8,  #0
    vcge.f32 q3, q9,  #0
    vcge.f32 q4, q10, #0
    vcge.f32 q5, q11, #0
    vbsl.f32 q2, q0, q1
    vbsl.f32 q3, q0, q1
    vbsl.f32 q4, q0, q1
    vbsl.f32 q5, q0, q1

    vadd.f32 q8, q8, q2
    vcge.f32 q2, q12, #0
    vadd.f32 q9, q9, q3
    vcge.f32 q3, q13, #0
    vadd.f32 q10, q10, q4
    vcge.f32 q4, q14, #0
    vadd.f32 q11, q11, q5
    vcge.f32 q5, q15, #0

    vbsl.f32 q2, q0, q1
    vbsl.f32 q3, q0, q1
    vbsl.f32 q4, q0, q1
    vbsl.f32 q5, q0, q1

    vadd.f32 q12, q12, q2
    vadd.f32 q13, q13, q3
    vadd.f32 q14, q14, q4
    vadd.f32 q15, q15, q5

    vcvt.s32.f32 q8,  q8
    vcvt.s32.f32 q9,  q9
    vcvt.s32.f32 q10, q10
    vcvt.s32.f32 q11, q11
    vcvt.s32.f32 q12, q12
    vcvt.s32.f32 q13, q13
    vcvt.s32.f32 q14, q14
    vcvt.s32.f32 q15, q15

    vqmovn.s32 d16, q8
    vqmovn.s32 d17, q9
    vqmovn.s32 d18, q10
    vqmovn.s32 d19, q11
    vqmovn.s32 d20, q12
    vqmovn.s32 d21, q13
    vqmovn.s32 d22, q14
    vqmovn.s32 d23, q15

    vqmovn.s16 d0, q8
    vqmovn.s16 d1, q9
    vqmovn.s16 d2, q10
    vqmovn.s16 d3, q11

    cmp r7, #1   // if relu != 1 or 2, Conv-Add-Relu or Relu6, skip
    blt ConvAddPostEnd

    veor d4, d4, d4
    vmax.s8 d0, d0, d4
    vmax.s8 d1, d1, d4
    vmax.s8 d2, d2, d4
    vmax.s8 d3, d3, d4

    cmp r7, #2   // relu6
    bne ConvAddPostEnd
    ldr r8, [sp, #128]  // relu6_max
    vldr d4, [r8]
    vmin.s8 d0, d0, d4
    vmin.s8 d1, d1, d4
    vmin.s8 d2, d2, d4
    vmin.s8 d3, d3, d4

ConvAddPostEnd:
    // store to dst_ptr 0 ~ 3
    add r9, r0, r4
    add r10, r0, r4, lsl#1
    add r11, r9, r4, lsl#1

    vstr d0, [r0]
    vstr d1, [r9]
    vstr d2, [r10]
    vstr d3, [r11]

    // dst_ptr += 4 * dst_depth
    add r0, r0, r4, lsl#2

    b LoopHW4

LoopHW1:
    cmp r5, #0
    ble LoopHW1End

    // src_ptr 0
    mov r9,  r1

    // load bias 32bytes, accumulator 8 (hw4 oc8) reg
    vld1.32 {q0, q1}, [r6]

    // src_depth counter
    mov r7, r3

    // weight_ptr
    mov r8, r2

    HW1LoopCrr16:
        cmp r7, #15
        ble HW1LoopCrr8

        vld1.8 {q2}, [r9]!

        vld1.8 {q8, q9}, [r8]!
        vld1.8 {q10, q11}, [r8]!
        vld1.8 {q12, q13}, [r8]!
        vld1.8 {q14, q15}, [r8]!

        sub r7, #16
        .word 0xfe200dc4 // vsdot.s8 q0, q8,  d4[0]
        .word 0xfe222dc4 // vsdot.s8 q1, q9,  d4[0]
        .word 0xfe240de4 // vsdot.s8 q0, q10, d4[1]
        .word 0xfe262de4 // vsdot.s8 q1, q11, d4[1]
        .word 0xfe280dc5 // vsdot.s8 q0, q12, d5[0]
        .word 0xfe2a2dc5 // vsdot.s8 q1, q13, d5[0]
        .word 0xfe2c0de5 // vsdot.s8 q0, q14, d5[1]
        .word 0xfe2e2de5 // vsdot.s8 q1, q15, d5[1]
        b HW1LoopCrr16

    HW1LoopCrr8:
        cmp r7, #7
        ble HW1LoopCrr4

        vld1.8 {d4}, [r9]!

        vld1.8 {q8, q9}, [r8]!
        vld1.8 {q10, q11}, [r8]!

        sub r7, #8
        .word 0xfe200dc4 // vsdot.s8 q0, q8,  d4[0]
        .word 0xfe222dc4 // vsdot.s8 q1, q9,  d4[0]
        .word 0xfe240de4 // vsdot.s8 q0, q10, d4[1]
        .word 0xfe262de4 // vsdot.s8 q1, q11, d4[1]
        b HW1LoopCrr8

    HW1LoopCrr4:
        cmp r7, #3
        ble HW1LoopEnd

        vld1.32 {d4[0]}, [r9]!
        vld1.8 {q8, q9}, [r8]!
        sub r7, #4
        .word 0xfe200dc4 // vsdot.s8 q0, q8,  d4[0]
        .word 0xfe222dc4 // vsdot.s8 q1, q9,  d4[0]
        b HW1LoopCrr4

HW1LoopEnd:
    // hw counter -= 1
    sub r5, #1
    // src_ptr += 1 * src_depth
    add r1, r1, r3

    ldr r8, [sp, #112]  // scale_ptr
    ldr r7, [sp, #116]  // relu
    
    // scale oc0 ~ oc7
    vld1.32 {q8, q9}, [r8]

    ldr r8, [sp, #120]  // add_input

HW1ConvReluAdd:
    cmp r7, #-1   // if relu == -1, Conv-Relu-Add
    bne HW1MulScale

    veor q10, q10, q10
    vmax.s32 q0, q0, q10
    vmax.s32 q1, q1, q10
HW1MulScale:
    vcvt.f32.s32 q0, q0
    vcvt.f32.s32 q1, q1
    vmul.f32 q0, q0, q8
    vmul.f32 q1, q1, q9

    cmp r8, #0       // if add_input == 0, skip
    beq HW1ConvAddPost

HW1AddInputScale:
    ldr r12, [sp, #124]   // add_scale

    vldr d16, [r8]
    // add_input_ptr += 1 * dst_depth
    add r8, r8, r4

    // add_scale
    vld1.32 {q14, q15}, [r12]

    // convert add_input int8 to fp32
    vmovl.s8 q8, d16
    vmovl.s16 q12, d16
    vmovl.s16 q13, d17
    vcvt.f32.s32 q12, q12
    vcvt.f32.s32 q13, q13

    vmla.f32 q0, q12, q14   // result += add_input * add_scale
    vmla.f32 q1, q13, q15

    str r8, [sp, #120]  // update add_input

HW1ConvAddPost:
    // f32 --> s32 --> s8
    // val + (val >= 0.f ? 0.5f : -0.5f)
    vmov.f32 q8, #0.5
    vmov.f32 q9, #-0.5

    vcge.f32 q10, q0, #0
    vcge.f32 q11, q1, #0
    vbsl.f32 q10, q8, q9
    vbsl.f32 q11, q8, q9
    vadd.f32 q0, q0, q10
    vadd.f32 q1, q1, q11
    vcvt.s32.f32 q0, q0
    vcvt.s32.f32 q1, q1

    vqmovn.s32 d16, q0
    vqmovn.s32 d17, q1
    vqmovn.s16 d0, q8

    cmp r7, #1   // if relu != 1 or 2, Conv-Add-Relu or Relu6, skip
    blt HW1ConvAddPostEnd

    veor d4, d4, d4
    vmax.s8 d0, d0, d4

    cmp r7, #2   // relu6
    bne HW1ConvAddPostEnd
    ldr r8, [sp, #128]  // relu6_max
    vldr d4, [r8]
    vmin.s8 d0, d0, d4

HW1ConvAddPostEnd:
    vstr d0, [r0]
    // dst_ptr += 1 * dst_depth
    add r0, r0, r4

    b LoopHW1

LoopHW1End:

vpop {q4-q7}
pop {r4-r11, pc}

END:

#endif
#endif
#endif
