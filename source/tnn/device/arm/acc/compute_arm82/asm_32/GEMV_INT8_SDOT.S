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

asm_function GemvInt8Sdot
//void GemvInt8Sdot(int8_t* dst, const int8_t* src, const int8_t* weight,
//                  const int32_t* bias, const float* scale, long ic_r4, long oc_r4)
//r0(dst),
//r1(src),
//r2(weight),
//r3(bias)

push {r4-r11, lr}
vpush {q4-q7}
// sp offset 9 x 4 + 16 x 4 = 100
//from stack (scale)   [sp, #100]
//from stack (ic_r4)   [sp, #104]
//from stack (oc_r4)   [sp, #108]

ldr r4, [sp, #100]
ldr r5, [sp, #104]
ldr r6, [sp, #108]

// cvt from fp32 to int8
vmov.f32 q5, #0.5
vmov.f32 q6, #-0.5

LoopOc16:
    cmp r6, #15
    ble LoopOc4

    sub r6, #16
    vldm r3!, {d0-d7}  // init from bias q0, q1, q2, q3

    mov r9, r1   // src_ptr
    mov r10, r5  // ic counter
    mov r11, r2  // weight_ptr

    Oc16L8:
        cmp r10, #7
        ble Oc16L4

        sub r10, #8
        vld1.8 {d8}, [r9]!
        vldm r11!, {d16-d31}    // weight

        .word 0xfe200dc8 // vsdot.s8 q0, q8,  d8[0]
        .word 0xfe222dc8 // vsdot.s8 q1, q9,  d8[0]
        .word 0xfe244dc8 // vsdot.s8 q2, q10, d8[0]
        .word 0xfe266dc8 // vsdot.s8 q3, q11, d8[0]
        .word 0xfe280de8 // vsdot.s8 q0, q12, d8[1]
        .word 0xfe2a2de8 // vsdot.s8 q1, q13, d8[1]
        .word 0xfe2c4de8 // vsdot.s8 q2, q14, d8[1]
        .word 0xfe2e6de8 // vsdot.s8 q3, q15, d8[1]
        b Oc16L8
    
    Oc16L4:
        cmp r10, #3
        ble Oc16L4End

        sub r10, #4
        vld1.32 {d8[0]}, [r9]!
        vldm r11!, {d16-d23}

        .word 0xfe200dc8 // vsdot.s8 q0, q8,  d8[0]
        .word 0xfe222dc8 // vsdot.s8 q1, q9,  d8[0]
        .word 0xfe244dc8 // vsdot.s8 q2, q10, d8[0]
        .word 0xfe266dc8 // vsdot.s8 q3, q11, d8[0]
        b Oc16L4

    Oc16L4End:
        vldm r4!, {d16-d23}  // scale

        vcvt.f32.s32 q0, q0
        vcvt.f32.s32 q1, q1
        vcvt.f32.s32 q2, q2
        vcvt.f32.s32 q3, q3
        vmul.f32 q0, q0, q8
        vmul.f32 q1, q1, q9
        vmul.f32 q2, q2, q10
        vmul.f32 q3, q3, q11

        vcge.f32 q8,  q0, #0
        vcge.f32 q9,  q1, #0
        vcge.f32 q10, q2, #0
        vcge.f32 q11, q3, #0
        vbsl.f32 q8,  q5, q6
        vbsl.f32 q9,  q5, q6
        vbsl.f32 q10, q5, q6
        vbsl.f32 q11, q5, q6

        vadd.f32 q0, q0, q8
        vadd.f32 q1, q1, q9
        vadd.f32 q2, q2, q10
        vadd.f32 q3, q3, q11

        vcvt.s32.f32 q0, q0
        vcvt.s32.f32 q1, q1
        vcvt.s32.f32 q2, q2
        vcvt.s32.f32 q3, q3

        vqmovn.s32 d0, q0
        vqmovn.s32 d1, q1
        vqmovn.s32 d2, q2
        vqmovn.s32 d3, q3

        vqmovn.s16 d0, q0
        vqmovn.s16 d1, q1

        vst1.8 {q0}, [r0]!
        // weight += 16 * ic_r4
        add r2, r2, r5, lsl#4

        b LoopOc16

LoopOc4:
    cmp r6, #3
    ble END

    sub r6, #4
    vld1.8 {q0}, [r3]!

    mov r9, r1   // src_ptr
    mov r10, r5  // ic counter
    mov r11, r2  // weight_ptr

    Oc4L16:
        cmp r10, #15
        ble Oc4L4

        sub r10, #16
        vld1.8 {q4}, [r9]!
        vldm r11!, {d16-d23}    // weight

        .word 0xfe200dc8 // vsdot.s8 q0, q8,  d8[0]
        .word 0xfe220de8 // vsdot.s8 q0, q9,  d8[1]
        .word 0xfe240dc9 // vsdot.s8 q0, q10, d9[0]
        .word 0xfe260de9 // vsdot.s8 q0, q11, d9[1]

        b Oc4L16
    
    Oc4L4:
        cmp r10, #3
        ble Oc4L4End

        sub r10, #4
        vld1.32 {d8[0]}, [r9]!
        vld1.8 {q8}, [r11]!

        .word 0xfe200dc8 // vsdot.s8 q0, q8, d8[0]
        b Oc4L4
    
    Oc4L4End:
        vld1.8 {q8}, [r4]!     // scale
        vcvt.f32.s32 q0, q0
        vmul.f32 q0, q0, q8

        vcge.f32 q8, q0, #0
        vbsl.f32 q8, q5, q6
        vadd.f32 q0, q0, q8
        vcvt.s32.f32 q0, q0
        vqmovn.s32 d0, q0
        vqmovn.s16 d0, q0

        vst1.32 {d0[0]}, [r0]!
        // weight += 4 * ic_r4
        add r2, r2, r5, lsl#2

        b LoopOc4

END:
vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif
#endif
