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

asm_function GemmFp16SlidewC3
//void GemmFp16SlidewC3(__fp16* dst,            //x0: dst 
//                          const __fp16* src,      //x1: src
//                          const __fp16* weight,   //x2: weight
//                          int width,             //x3: width
//                          int src_w_setup,       //x4: src_w_step
//                          int fw,                //x5: fw
//                          int fh,                //x6: fh
//                          int dilateX_step,      //x7: dilateX_step
//                          int dilateY_step);     //x8: dilateY_step, load from stack
dst          .req r0
src          .req r1
weight       .req r2
width        .req r3
src_w_step   .req r4
fw           .req r7
fh           .req r8
dilateX_step .req r9
dilate_y_gap .req r10

push {r4-r11, lr}

//Auto Load:
//r0:dst, r1:src, r2:weight, r3:width

//Load from sp
//r4:src_w_step, r7:fw, r8:fh, r9:dilate_x_step, r10:dilate_y_step
ldr r4, [sp, #36]
ldr r7, [sp, #40]
ldr r8, [sp, #44]
ldr r9, [sp, #48]
ldr r10, [sp, #52]

vpush {q4-q7}

//step multi by sizeof(float)
mov r12, #2
mul r10, r12, r10
mul r9, r12, r9
mul r4, r12, r4

//dilate_y_step -> dilate_y_step-fw*dilate_x_step
mul r12, r7, r9
sub r10, r10, r12

L8:
cmp r3, #7
ble L4


L8Loop:
    mov r5, src
    mov r6, weight
    vmov.i32 d14[0], width
    mov width, #8
    mul width, src_w_step, width
    vmov.i16 q8,  #0
    vmov.i16 q9,  #0
    vmov.i16 q10, #0
    vmov.i16 q11, #0
    vmov.i16 q12, #0
    vmov.i16 q13, #0
    vmov.i16 q14, #0
    vmov.i16 q15, #0
    mov r11, fh
    L8LoopFY:
        mov r12, fw
        L8LoopFX:
            vld1.16 {q4, q5}, [weight]!
            vld1.16 {q6}, [weight]!

            vld1.16 {q0}, [src], src_w_step
            vld1.16 {q1}, [src], src_w_step

            vmla.f16 q8, q4, d0[0]
            vmla.f16 q8, q5, d0[1]
            vmla.f16 q9, q4, d2[0]

            vld1.16 {q2}, [src], src_w_step

            vmla.f16 q8, q6, d0[2]
            vmla.f16 q9, q5, d2[1]
            vmla.f16 q10, q4, d4[0]

            vld1.16 {q3}, [src], src_w_step

            vmla.f16 q9, q6, d2[2]
            vmla.f16 q10, q5, d4[1]
            vmla.f16 q11, q5, d6[1]
            
            vld1.16 {q0}, [src], src_w_step
            vmla.f16 q11, q6, d6[2]
            vmla.f16 q10, q6, d4[2]
            vmla.f16 q11, q4, d6[0]


            vld1.16 {q1}, [src], src_w_step
            vmla.f16 q12, q4, d0[0]
            vmla.f16 q12, q5, d0[1]
            vmla.f16 q13, q4, d2[0]
            
            vld1.16 {q2}, [src], src_w_step

            vmla.f16 q12, q6, d0[2]
            vmla.f16 q13, q5, d2[1]
            vmla.f16 q14, q4, d4[0]

            vld1.16 {q3}, [src], src_w_step
            vmla.f16 q13, q6, d2[2]
            vmla.f16 q14, q5, d4[1]
            vmla.f16 q15, q5, d6[1]
            vmla.f16 q15, q6, d6[2]
            vmla.f16 q14, q6, d4[2]
            vmla.f16 q15, q4, d6[0]

            sub src, src, width
            subs fw, fw, #1
            add src, src, dilateX_step
            bne L8LoopFX
        subs fh, fh, #1
        mov fw, r12
        add src, src, dilate_y_gap
        bne L8LoopFY
    mov fh, r11
    mov src, r5
    add src, src, width
    mov weight, r6
    vmov.i32 width, d14[0]
    vst1.16 {q8, q9}, [dst]!
    vst1.16 {q10, q11}, [dst]!
    sub width, width, #8
    vst1.16 {q12, q13}, [dst]!
    cmp width, #8
    vst1.16 {q14, q15}, [dst]!
    bge L8Loop

L4:
cmp width, #3
ble L1


L4Loop:
    mov r5, src
    mov r6, weight
    vmov.i16 d14[0], width
    mov width, #4
    mul width, src_w_step, width
    vmov.i16 q8, #0
    vmov.i16 q9, #0
    vmov.i16 q10, #0
    vmov.i16 q11, #0
    mov r11, fh
    L4LoopFY:
        mov r12, fw
        L4LoopFX:
            vld1.16 {q4, q5}, [weight]!
            vld1.16 {q6}, [weight]!

            vld1.16 {q0}, [src], src_w_step
            vld1.16 {q1}, [src], src_w_step

            vmla.f16 q8, q4, d0[0]
            vmla.f16 q8, q5, d0[1]
            vmla.f16 q9, q4, d2[0]

            vld1.16 {q2}, [src], src_w_step
            vmla.f16 q8, q6, d0[2]
            vmla.f16 q9, q5, d2[1]
            vmla.f16 q10, q4, d4[0]

            vld1.16 {q3}, [src], src_w_step

            vmla.f16 q9, q6, d2[2]
            vmla.f16 q10, q5, d4[1]
            vmla.f16 q11, q5, d6[1]
            vmla.f16 q11, q6, d6[2]
            vmla.f16 q10, q6, d4[2]
            vmla.f16 q11, q4, d6[0]

            sub src, src, width
            subs fw, fw, #1
            add src, src, dilateX_step
            bne L4LoopFX
        subs fh, fh, #1
        mov fw, r12
        add src, src, dilate_y_gap
        bne L4LoopFY
    mov fh, r11
    mov src, r5
    add src, src, width
    mov weight, r6
    vst1.16 {q8, q9}, [dst]!
    vmov.i32 width, d14[0]
    sub width, width, #4
    vst1.16 {q10, q11}, [dst]!


L1:
cmp width, #0
ble End

L1Loop:
    mov r5, src
    mov r6, weight
    vmov.i16 q0, #0
    vmov.i16 q1, #0
    mov r11, fh
    L1LoopFY:
        mov r12, fw
        L1LoopFX:
            vld1.16 {q3}, [src], dilateX_step
            vld1.16 {q4, q5}, [weight]!
            vmla.f16 q0, q4, d6[0]
            vmla.f16 q1, q5, d6[1]
            vld1.16 {q6}, [weight]!
            vmla.f16 q0, q6, d6[2]
            subs fw, fw, #1
            bne L1LoopFX
        subs fh, fh, #1
        mov fw, r12
        add src, src, dilate_y_gap
        bne L1LoopFY
    mov fh, r11
    vadd.f16 q0, q0, q1
    mov src, r5
    mov weight, r6
    add src, src, src_w_step
    vst1.16 {q0}, [dst]!
    subs width, width, #1
    bne L1Loop

End:

vpop {q4-q7}
pop {r4-r11, pc}


#endif
#endif
#endif
