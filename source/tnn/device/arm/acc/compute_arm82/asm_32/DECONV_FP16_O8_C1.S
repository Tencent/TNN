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

asm_function DeconvFp16O8C1
//input is nchw
//output is nc8hw8
//void DeconvFp16O8C1(__fp16* dst,             // r0
//                     const __fp16* src,    // r1
//                     const __fp16* weight, // r2
//                     int width,            // r3
//                     int dst_w_step,       // r4
//                     int src_depth_div8,   // r5
//                     int src_depth_step,   // r6
//                     int fw,               // r7
//                     int fh,               // r8
//                     int dilate_x_step,    // r9
//                     int dilate_y_step)    // r10

dst          .req r0
src          .req r1
weight       .req r2
width        .req r3
dst_w_step   .req r4
ic           .req r5
fw           .req r7
fh           .req r8
dilate_x_step .req r9
dilate_y_step .req r10

push {r4-r11, lr}

//Auto Load:
//r0:dst, r1:src, r2:weight, r3:width

//Load from sp
//r4:dst_w_step, r5:src_depth_div8, r6: src_depth_step
//r7:fw, r8:fh, r9:dilate_x_step, r10:dilate_y_step
ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
ldr r7, [sp, #48]
ldr r8, [sp, #52]
ldr r9, [sp, #56]
ldr r10, [sp, #60]

vpush {q4-q7}

//step multi by sizeof(__fp16)
lsl r10, r10, #1
lsl r9, r9, #1
lsl r6, r6, #1
lsl r4, r4, #1

//src_depth_step -> src_depth_step - fh*dilate_y_step
//mul x12, x8, x10
//sub x6, x6, x12

//dilate_y_step -> dilate_y_step-fw*dilate_x_step
//mul x12, x7, x9
//sub x10, x10, x12

L8:
cmp r3, #7
ble L4

L8Loop:
    vmov.i32 d8[0], src
    vmov.i32 d8[1], weight
    vmov.i32 d9[0], width

    lsl width, dst_w_step, #3

    mov r11, fh
    L8LoopFY:
        mov r12, fw
        L8LoopFX:
            vmov.i32 d9[1], ic
            vld1.16 {q7}, [weight]!
            vld1.16 {q0}, [src], r6
            vmul.f16 q8,  q7, d0[0]
            vmul.f16 q9,  q7, d0[1]
            vmul.f16 q10, q7, d0[2]
            vmul.f16 q11, q7, d0[3]
            vmul.f16 q12, q7, d1[0]
            vmul.f16 q13, q7, d1[1]
            vmul.f16 q14, q7, d1[2]
            vmul.f16 q15, q7, d1[3]

            subs ic, ic, #1
            beq L8LoopZEnd
            L8LoopZ:
                vld1.16 {q7}, [weight]!
                vld1.16 {q0}, [src], r6

                subs ic, ic, #1
                vmla.f16 q8,  q7, d0[0]
                vmla.f16 q9,  q7, d0[1]
                vmla.f16 q10, q7, d0[2]
                vmla.f16 q11, q7, d0[3]
                vmla.f16 q12, q7, d1[0]
                vmla.f16 q13, q7, d1[1]
                vmla.f16 q14, q7, d1[2]
                vmla.f16 q15, q7, d1[3]

                bne L8LoopZ
            L8LoopZEnd:

            vld1.16 {q0}, [dst], dst_w_step
            vld1.16 {q1}, [dst], dst_w_step
            vld1.16 {q2}, [dst], dst_w_step
            vld1.16 {q3}, [dst], dst_w_step

            // add with stride
            vadd.f16 q8,  q8,  q0
            vld1.16 {q0}, [dst], dst_w_step
            vadd.f16 q9,  q9,  q1
            vld1.16 {q1}, [dst], dst_w_step
            vadd.f16 q10, q10, q2
            vld1.16 {q2}, [dst], dst_w_step
            vadd.f16 q11, q11, q3
            vld1.16 {q3}, [dst], dst_w_step

            sub dst, dst, width
            vadd.f16 q12, q12, q0
            vadd.f16 q13, q13, q1
            vadd.f16 q14, q14, q2
            vadd.f16 q15, q15, q3

            vst1.16 {q8},  [dst], dst_w_step
            vst1.16 {q9},  [dst], dst_w_step
            vst1.16 {q10}, [dst], dst_w_step
            vst1.16 {q11}, [dst], dst_w_step
            vst1.16 {q12}, [dst], dst_w_step
            vst1.16 {q13}, [dst], dst_w_step
            vst1.16 {q14}, [dst], dst_w_step
            vst1.16 {q15}, [dst], dst_w_step
            sub dst, dst, width
            add dst, dst, dilate_x_step

            vmov.i32 ic, d9[1]
            subs fw, fw, #1
            vmov.i32 src, d8[0]
            bne L8LoopFX
        subs fh, fh, #1
        mov fw, r12
        mul r12, fw, dilate_x_step
        sub dst, dst, r12
        add dst, dst, dilate_y_step
        bne L8LoopFY

    mov fh, r11
    mul r12, fh, dilate_y_step
    sub dst, dst, r12
    add src, src, #16
    add dst, dst, width
    vmov.i32 weight, d8[1]
    vmov.i32 width, d9[0]
    sub width, width, #8
    cmp width, #8
    bge L8Loop

L4:
cmp r3, #3
ble L1

L4Loop:
    vmov.i32 d8[0], src
    vmov.i32 d8[1], weight
    vmov.i32 d9[0], width

    lsl width, dst_w_step, #2

    mov r11, fh
    L4LoopFY:
        mov r12, fw
        L4LoopFX:
            vmov.i32 d9[1], ic
            vld1.16 {q12}, [weight]!
            vld1.16 {d0}, [src], r6
            vmul.f16 q8,  q12, d0[0]
            vmul.f16 q9,  q12, d0[1]
            vmul.f16 q10, q12, d0[2]
            vmul.f16 q11, q12, d0[3]

            subs ic, ic, #1
            beq L4LoopZEnd
            L4LoopZ:
                vld1.16 {q12}, [weight]!
                vld1.16 {d0}, [src], r6

                vmla.f16 q8,  q12, d0[0]
                vmla.f16 q9,  q12, d0[1]
                vmla.f16 q10, q12, d0[2]
                vmla.f16 q11, q12, d0[3]

                subs ic, ic, #1
                bne L4LoopZ

            L4LoopZEnd:

            vld1.16 {q0}, [dst], dst_w_step
            vld1.16 {q1}, [dst], dst_w_step
            vld1.16 {q2}, [dst], dst_w_step
            vld1.16 {q3}, [dst], dst_w_step
            sub dst, dst, width

            // add with stride
            vadd.f16 q8,  q8,  q0
            vadd.f16 q9,  q9,  q1
            vadd.f16 q10, q10, q2
            vadd.f16 q11, q11, q3

            vst1.16 {q8},  [dst], dst_w_step
            vst1.16 {q9},  [dst], dst_w_step
            vst1.16 {q10}, [dst], dst_w_step
            vst1.16 {q11}, [dst], dst_w_step
            sub dst, dst, width
            add dst, dst, dilate_x_step

            vmov.i32 ic, d9[1]
            subs fw, fw, #1
            vmov.i32 src, d8[0] 
            bne L4LoopFX
        subs fh, fh, #1
        mov fw, r12
        mul r12, fw, dilate_x_step
        sub dst, dst, r12
        add dst, dst, dilate_y_step
        bne L4LoopFY
    
    mov fh, r11
    mul r12, fh, dilate_y_step
    sub dst, dst, r12
    add src, src, #8
    add dst, dst, width
    vmov.i32 weight, d8[1]
    vmov.i32 width, d9[0]
    sub width, width, #4
    cmp width, #4
    bge L4Loop

L1:
cmp r3, #0
ble End

L1Loop:
    vmov.i32 d8[0], src
    vmov.i32 d8[1], weight
    vmov.i32 d9[0], width

    mov r11, fh
    L1LoopFY:
        mov r12, fw
        L1LoopFX:
            vmov.i32 d9[1], ic
            veor q8,  q8,  q8
            L1LoopZ:
                vld1.16 {q1}, [weight]!
                vld1.16 {d0[0]}, [src], r6

                vmla.f16 q8,  q1,  d0[0]

                subs ic, ic, #1
                bne L1LoopZ

            L1LoopZEnd:
            vld1.16 {q0}, [dst]
            // add with stride
            vadd.f16 q8,  q8,  q0
            vst1.16 {q8}, [dst]

            add dst, dst, dilate_x_step

            vmov.i32 ic, d9[1]
            subs fw, fw, #1
            vmov.i32 src, d8[0]
            bne L1LoopFX
        subs fh, fh, #1
        mov fw, r12
        mul r12, fw, dilate_x_step
        sub dst, dst, r12
        add dst, dst, dilate_y_step
        bne L1LoopFY

    mov fh, r11
    mul r12, fh, dilate_y_step
    sub dst, dst, r12
    add src, src, #2
    add dst, dst, dst_w_step
    vmov.i32 weight, d8[1]
    vmov.i32 width, d9[0]
    sub width, width, #1
    cmp width, #1
    bge L1Loop

End:

vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif
#endif
