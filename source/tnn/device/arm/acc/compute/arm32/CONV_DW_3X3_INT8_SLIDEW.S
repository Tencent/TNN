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

//void DepthwiseI8K3S1Kernel(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias_z, long width,
//                           long src_y_step, long src_w_step, long dst_depth, long fw, long fh, const float* scale_z,
//                           long dx, long dc)

        push    {r4, r5, r6, r7, r8, r9, r10, lr}
        vpush.64        {d8, d9, d10, d11, d12, d13, d14, d15}
        sub     sp, sp, #16
        ldr     ip, [sp, #144]
        ldr     r7, [sp, #140]
        lsl     r5, ip, #2
        add     r9, r5, #16
        add     lr, r3, r9
        add     r3, r3, r5
        vld1.32 {d20-d21}, [lr]
        vld1.32 {d4-d5}, [r3]
        vmov    q5, q10  @ v4si
        vmov    q1, q10  @ v4si
        vmov    q0, q10  @ v4si
        vmov    q4, q2  @ v4si
        vmov    q8, q2  @ v4si
        vmov    q9, q2  @ v4si
        mov     lr, #3
        ldr     r6, [sp, #120]
        ldr     r4, [sp, #124]
        mla     r3, r7, r6, ip
        ldr     r8, [sp, #116]
        mla     r6, r4, r7, ip
        ldr     r7, [sp, #136]
        add     r2, r2, ip
        add     r1, r1, r3
        lsl     r10, r4, #1
.L2:
        mov     r3, r1
        vld1.8  {d22}, [r3], r4
        vld1.8  {d30}, [r3], r4
        vld1.8  {d26}, [r2], r4
        vld1.8  {d6}, [r3]
        add     ip, r3, r4
        vmovl.s8        q13, d26
        vmovl.s8        q3, d6
        vmovl.s8        q15, d30
        vld1.8  {d12}, [ip], r4
        vstr    d22, [sp]
        vmovl.s8        q6, d12
        vmov    d22, d27  @ v4hi
        vmov    d27, d6  @ v4hi
        vmov    d6, d30  @ v4hi
        vldr    d30, [sp]
        vmov    d29, d12  @ v4hi
        vmov    d28, d13  @ v4hi
        vld1.8  {d14}, [r2], r4
        vmovl.s8        q6, d30
        vld1.8  {d24}, [ip], r4
        vmov    d23, d26  @ v4hi
        vmovl.s8        q12, d24
        vmovl.s8        q7, d14
        vmov    d26, d7  @ v4hi
        vmlal.s16       q10, d13, d22
        vld1.8  {d7}, [r2], r4
        vmlal.s16       q2, d12, d23
        vstr    d7, [sp, #8]
        vmov    q6, q10  @ v4si
        vmov    d7, d31  @ v4hi
        vmov    d31, d24  @ v4hi
        vmov    d24, d15  @ v4hi
        vmov    d30, d25  @ v4hi
        vld1.8  {d21}, [ip]
        vmov    d25, d14  @ v4hi
        vmlal.s16       q6, d7, d24
        vmlal.s16       q4, d29, d23
        vmlal.s16       q5, d28, d22
        vmlal.s16       q8, d27, d23
        vmlal.s16       q1, d26, d22
        vldr    d14, [sp, #8]
        vmlal.s16       q9, d6, d23
        vmlal.s16       q0, d7, d22
        vmovl.s8        q7, d14
        vmlal.s16       q4, d31, d25
        vmlal.s16       q5, d30, d24
        vmlal.s16       q8, d29, d25
        vmlal.s16       q1, d28, d24
        vmlal.s16       q9, d27, d25
        vmlal.s16       q0, d26, d24
        vmlal.s16       q2, d6, d25
        vmovl.s8        q12, d21
        vmov    q10, q6  @ v4si
        subs    lr, lr, #1
        add     r1, r1, r8
        vmlal.s16       q8, d31, d14
        vmlal.s16       q1, d30, d15
        vmlal.s16       q9, d29, d14
        vmlal.s16       q0, d28, d15
        vmlal.s16       q2, d27, d14
        vmlal.s16       q10, d26, d15
        vmlal.s16       q4, d24, d14
        vmlal.s16       q5, d25, d15
        bne     .L2
        add     r5, r7, r5
        add     r7, r7, r9
        vld1.32 {d28-d29}, [r7]
        vcvt.f32.s32    q10, q10
        ldr     r3, .L6
        vld1.32 {d30-d31}, [r5]
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
        vmov    d5, d20  @ v4hi
        vqmovn.s32      d0, q0
        vqmovn.s32      d28, q14
        add     r0, r0, r6
        mov     r3, r0
        vqmovn.s32      d24, q12
        vqmovn.s32      d22, q11
        vqmovn.s32      d18, q9
        vqmovn.s16      d4, q2
        vmov    d25, d0  @ v4hi
        vmov    d23, d28  @ v4hi
        vqmovn.s32      d16, q8
        vst1.8  {d4}, [r3], r4
        vmov    d17, d16  @ v4hi
        vqmovn.s16      d24, q12
        vqmovn.s16      d22, q11
        vmov    d16, d18  @ v4hi
        add     r10, r0, r10
        vst1.8  {d24}, [r3]
        vqmovn.s16      d16, q8
        vst1.8  {d22}, [r10], r4
        vst1.8  {d16}, [r10]
        add     sp, sp, #16
        vldm    sp!, {d8-d15}
        pop     {r4, r5, r6, r7, r8, r9, r10, pc}

.L6:
    .word 0x4B400000

#endif
#endif
