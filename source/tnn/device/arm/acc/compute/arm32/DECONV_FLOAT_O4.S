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

asm_function DeconvFloatO4
//void DeconvFloatO4(float* dst,           // r0
//                     const float* src,     // r1
//                     const float* weight,  // r2
//                     int width,            // r3
//                     int dst_w_step,       // r4
//                     int src_depth_quad,   // r5
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
ic4          .req r5
fw           .req r7
fh           .req r8
dilate_x_step .req r9
dilate_y_step .req r10

push {r4-r11, lr}

//Auto Load:
//r0:dst, r1:src, r2:weight, r3:width

//Load from sp
//r4:dst_w_step, r5:src_depth_quad, r6: src_depth_step, r7:fw, r8:fh, r9:dilate_x_step, r10:dilate_y_step
ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
ldr r7, [sp, #48]
ldr r8, [sp, #52]
ldr r9, [sp, #56]
ldr r10, [sp, #60]

vpush {q4-q7}

//step multi by sizeof(float)
mov r12, #4
mul r10, r12, r10
mul r9, r12, r9
mul r6, r12, r6
mul dst_w_step, r12, dst_w_step

//src_depth_step -> src_depth_step - fh*dilate_y_step
//mul r12, fh, r10
//sub r6, r6, r12

//dilate_y_step -> dilate_y_step-fw*dilate_x_step
//mul r12, fw, r9
//sub r10, r10, r12

L4:
cmp r3, #3
ble L1


L4Loop:
    vmov.i32 d6[0], src
    vmov.i32 d6[1], weight
    vmov.i32 d7[0], width

    mov r3, #4
    mul r3, r3, dst_w_step

    mov r11, fh
    L4LoopFY:
        mov r12, fw
        L4LoopFX:
            vmov.i32 d7[1], ic4 
            vld1.32 {q12, q13}, [weight]!
            vld1.32 {q14, q15}, [weight]!

            vld1.32 {q0}, [src]!
            vld1.32 {q1}, [src]!
            vld1.32 {q2}, [src]!
            vld1.32 {q4}, [src]!
            vmul.f32 q8, q12, d0[0]
            vmul.f32 q9, q12, d2[0]
            vmul.f32 q10, q12, d4[0]
            vmul.f32 q11, q12, d8[0]

            subs ic4, ic4, #1
            beq L4LoopZEnd
            L4LoopZ:
                sub src, src, #64
                add src, src, r6
                vmla.f32 q8, q13, d0[1]
                vmla.f32 q9, q13, d2[1]
                vmla.f32 q10, q13, d4[1]
                vmla.f32 q11, q13, d8[1]

                vmla.f32 q8, q14, d1[0]
                vmla.f32 q9, q14, d3[0]
                vmla.f32 q10, q14, d5[0]
                vmla.f32 q11, q14, d9[0]

                vmla.f32 q8, q15, d1[1]
                vmla.f32 q9, q15, d3[1]
                vmla.f32 q10, q15, d5[1]
                vmla.f32 q11, q15, d9[1]

                vld1.32 {q12, q13}, [weight]!
                vld1.32 {q14, q15}, [weight]!

                vld1.32 {q0}, [src]!
                vld1.32 {q1}, [src]!
                vld1.32 {q2}, [src]!
                vld1.32 {q4}, [src]!

                vmla.f32 q8, q12, d0[0]
                vmla.f32 q9, q12, d2[0]
                vmla.f32 q10, q12, d4[0]
                vmla.f32 q11, q12, d8[0]

                subs ic4, ic4, #1
                bne L4LoopZ
            L4LoopZEnd:
            vmla.f32 q8, q13, d0[1]
            vmla.f32 q9, q13, d2[1]
            vmla.f32 q10, q13, d4[1]
            vmla.f32 q11, q13, d8[1]

            vmla.f32 q8, q14, d1[0]
            vmla.f32 q9, q14, d3[0]
            vmla.f32 q10, q14, d5[0]
            vmla.f32 q11, q14, d9[0]

            vmla.f32 q8, q15, d1[1]
            vmla.f32 q9, q15, d3[1]
            vmla.f32 q10, q15, d5[1]
            vmla.f32 q11, q15, d9[1]

            // add with stride
            vld1.32 {q0}, [r0]
            vadd.f32 q8, q8, q0
            vst1.32 {q8}, [r0], dst_w_step

            vld1.32 {q0}, [r0]
            vadd.f32 q9, q9, q0
            vst1.32 {q9}, [r0], dst_w_step

            vld1.32 {q0}, [r0]
            vadd.f32 q10, q10, q0
            vst1.32 {q10}, [r0], dst_w_step

            vld1.32 {q0}, [r0]
            vadd.f32 q11, q11, q0
            vst1.32 {q11}, [r0], dst_w_step

            sub r0, r0, r3
            add r0, r0, dilate_x_step

            vmov.i32 ic4, d7[1]
            subs fw, fw, #1
            vmov.i32 src, d6[0]
            bne L4LoopFX
        subs fh, fh, #1
        mov fw, r12
        mul r12, fw, dilate_x_step
        sub r0, r12
        add r0, r0, dilate_y_step 
        bne L4LoopFY
    mov fh, r11
    mul r12, fh, dilate_y_step
    sub r0, r0, r12
    add src, src, #64
    add r0, r0, r3
    vmov.i32 weight, d6[1]
    vmov.i32 r3, d7[0]
    sub r3, r3, #4
    cmp r3, #4
    bge L4Loop


L1:
cmp r3, #0
ble End

L1Loop:
    vmov.i32 d6[0], src
    vmov.i32 d6[1], weight
    vmov.i32 d7[0], width

    mov r3, #1
    mul r3, r3, dst_w_step

    mov r11, fh
    L1LoopFY:
        mov r12, fw
        L1LoopFX:
            vmov.i32 d7[1], ic4 
            vmov.i32 q8,  #0
            vmov.i32 q9,  #0
            L1LoopZ:
                vld1.32 {q0}, [src]!
                vld1.32 {q4, q5}, [weight]!
                vmla.f32 q8, q4, d0[0]
                vmla.f32 q9, q5, d0[1]
                vld1.32 {q6, q7}, [weight]!
                vmla.f32 q8, q6, d1[0]
                vmla.f32 q9, q7, d1[1]

                subs ic4, ic4, #1
                sub src, src, #16
                add src, src, r6
                bne L1LoopZ
            L1LoopZEnd:
            // add with stride
            vadd.f32 q8, q8, q9
            vld1.32 {q0}, [r0]
            vadd.f32 q8, q8, q0
            vst1.32 {q8}, [r0], dst_w_step

            sub r0, r0, r3
            add r0, r0, dilate_x_step

            vmov.i32 ic4, d7[1]
            subs fw, fw, #1
            vmov.i32 src, d6[0]
            bne L1LoopFX
        subs fh, fh, #1
        mov fw, r12
        mul r12, fw, dilate_x_step
        sub r0, r12
        add r0, r0, dilate_y_step 
        bne L1LoopFY
    mov fh, r11
    mul r12, fh, dilate_y_step
    sub r0, r0, r12
    add src, src, #16
    add r0, r0, r3
    vmov.i32 weight, d6[1]
    vmov.i32 r3, d7[0]
    sub r3, r3, #1
    cmp r3, #1
    bge L1Loop

End:

vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif
