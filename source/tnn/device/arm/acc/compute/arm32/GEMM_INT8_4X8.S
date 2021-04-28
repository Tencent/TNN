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
asm_function GemmInt8Unit4x8
//void GemmInt8Unit4x8(long mr, long nr, long k, const int8_t* a, long a_stride, const void* w, int8_t* c, long c_stride,
//                     const float* scales, long relu, const int8_t* add_input, const float* add_scale, const int8_t* relu6_max)
//r0(mr),
//r1(nr),
//r2(k),
//r3(src),
//4  from stack(a_stride),
//5  from stack(weight),
//6  from stack(dst),
//7  from stack(c_stride)
//8  from stack(scale)
//9  from stack(relu)
//10 from stack(add_input)
//11 from stack(add_scale)
//12 from stack(relu6_max)

// |relu6_max|   <-- sp 132
// |add_scale|   <-- sp 128
// |add_input|   <-- sp 124
// |relu     |   <-- sp 120
// |scale    |   <-- sp 116
// |c_stride |   <-- sp 112
// |dst      |   <-- sp 108
// |weight   |   <-- sp 104
// |a_stride |   <-- sp 100
// |r4-r11,lr|   total 36
// |q4-q7    |   total 64 <-- sp

// r3-r6 a0-a3

push {r4-r11, lr}
vpush {q4-q7}

// load bias 32bit, accumulator 8 reg
ldr r7, [sp, #104] // weight
vldm r7!, {d16-d19}
vmov q10, q8
vmov q11, q9
vmov q12, q8
vmov q13, q9
vmov q14, q8
vmov q15, q9

ldr r6, [sp, #100] // a_stride

ldr r8, [sp, #108] // dst
ldr r9, [sp, #112] // c_stride
ldr r10, [sp, #116] // scale
ldr r11, [sp, #120] // relu

# a1
cmp r0, #2
add r4, r3, r6
movlo r4, r3
# a2
add r5, r4, r6
movls r5, r4
# a3
cmp r0, #4
add r6, r5, r6
movne r6, r5

subs r2, r2, #8
blo 1f

0:
    vld1.8 d9, [r7]!
    vmovl.s8 q4, d9
    vld1.8 d1, [r3]!
    vmovl.s8 q0, d1
    vld1.8 d3, [r4]!
    vmovl.s8 q1, d3
    vld1.8 d5, [r5]!
    vmovl.s8 q2, d5
    vld1.8 d7, [r6]!
    vmovl.s8 q3, d7

    // c0
    vld1.8 d11, [r7]!
    vmlal.s16 q8, d8, d0[0]
    vmlal.s16 q9, d9, d0[0]
    vmlal.s16 q10, d8, d2[0]
    vmlal.s16 q11, d9, d2[0]
    vmovl.s8 q5, d11
    vmlal.s16 q12, d8, d4[0]
    vmlal.s16 q13, d9, d4[0]
    vmlal.s16 q14, d8, d6[0]
    vmlal.s16 q15, d9, d6[0]

    // c1
    vld1.8 d9, [r7]!
    vmlal.s16 q8, d10, d0[1]
    vmlal.s16 q9, d11, d0[1]
    vmlal.s16 q10, d10, d2[1]
    vmlal.s16 q11, d11, d2[1]
    vmovl.s8 q4, d9
    vmlal.s16 q12, d10, d4[1]
    vmlal.s16 q13, d11, d4[1]
    vmlal.s16 q14, d10, d6[1]
    vmlal.s16 q15, d11, d6[1]

    // c2
    vld1.8 d11, [r7]!
    vmlal.s16 q8, d8, d0[2]
    vmlal.s16 q9, d9, d0[2]
    vmlal.s16 q10, d8, d2[2]
    vmlal.s16 q11, d9, d2[2]
    vmovl.s8 q5, d11
    vmlal.s16 q12, d8, d4[2]
    vmlal.s16 q13, d9, d4[2]
    vmlal.s16 q14, d8, d6[2]
    vmlal.s16 q15, d9, d6[2]

    // c3
    vld1.8 d9, [r7]!
    vmlal.s16 q8, d10, d0[3]
    vmlal.s16 q9, d11, d0[3]
    vmlal.s16 q10, d10, d2[3]
    vmlal.s16 q11, d11, d2[3]
    vmovl.s8 q4, d9
    vmlal.s16 q12, d10, d4[3]
    vmlal.s16 q13, d11, d4[3]
    vmlal.s16 q14, d10, d6[3]
    vmlal.s16 q15, d11, d6[3]

    // c4
    vld1.8 d11, [r7]!
    vmlal.s16 q8, d8, d1[0]
    vmlal.s16 q9, d9, d1[0]
    vmlal.s16 q10, d8, d3[0]
    vmlal.s16 q11, d9, d3[0]
    vmovl.s8 q5, d11
    vmlal.s16 q12, d8, d5[0]
    vmlal.s16 q13, d9, d5[0]
    vmlal.s16 q14, d8, d7[0]
    vmlal.s16 q15, d9, d7[0]

    // c5
    vld1.8 d9, [r7]!
    vmlal.s16 q8, d10, d1[1]
    vmlal.s16 q9, d11, d1[1]
    vmlal.s16 q10, d10, d3[1]
    vmlal.s16 q11, d11, d3[1]
    vmovl.s8 q4, d9
    vmlal.s16 q12, d10, d5[1]
    vmlal.s16 q13, d11, d5[1]
    vmlal.s16 q14, d10, d7[1]
    vmlal.s16 q15, d11, d7[1]

    // c6
    vld1.8 d11, [r7]!
    vmlal.s16 q8, d8, d1[2]
    vmlal.s16 q9, d9, d1[2]
    vmlal.s16 q10, d8, d3[2]
    vmlal.s16 q11, d9, d3[2]
    vmovl.s8 q5, d11
    vmlal.s16 q12, d8, d5[2]
    vmlal.s16 q13, d9, d5[2]
    vmlal.s16 q14, d8, d7[2]
    vmlal.s16 q15, d9, d7[2]

    subs r2, r2, #8

    // c7
    vmlal.s16 q8, d10, d1[3]
    vmlal.s16 q9, d11, d1[3]
    vmlal.s16 q10, d10, d3[3]
    vmlal.s16 q11, d11, d3[3]
    vmlal.s16 q12, d10, d5[3]
    vmlal.s16 q13, d11, d5[3]
    vmlal.s16 q14, d10, d7[3]
    vmlal.s16 q15, d11, d7[3]

    bhs 0b

1:
    cmp r2, #-8
    beq 2f

    add r3, r3, r2
    add r4, r4, r2
    add r5, r5, r2
    add r6, r6, r2

    lsl r2, r2, #3
    vdup.32 d13, r2
    
    vld1.8 d1, [r3]!
    vld1.8 d3, [r4]!
    vld1.8 d5, [r5]!
    vld1.8 d7, [r6]!

    vshl.s64 d1, d1, d13
    vshl.s64 d3, d3, d13
    vshl.s64 d5, d5, d13
    vshl.s64 d7, d7, d13

    vmovl.s8 q0, d1
    vmovl.s8 q1, d3
    vmovl.s8 q2, d5
    vmovl.s8 q3, d7

    // c0
    vld1.8 d9, [r7]!
    vmovl.s8 q4, d9
    vmlal.s16 q8, d8, d0[0]
    vmlal.s16 q9, d9, d0[0]
    vmlal.s16 q10, d8, d2[0]
    vmlal.s16 q11, d9, d2[0]
    vmlal.s16 q12, d8, d4[0]
    vmlal.s16 q13, d9, d4[0]
    vmlal.s16 q14, d8, d6[0]
    vmlal.s16 q15, d9, d6[0]

    cmp r2, #-48
    blo 2f

    // c1
    vld1.8 d11, [r7]!
    vmovl.s8 q5, d11
    vmlal.s16 q8, d10, d0[1]
    vmlal.s16 q9, d11, d0[1]
    vmlal.s16 q10, d10, d2[1]
    vmlal.s16 q11, d11, d2[1]
    vmlal.s16 q12, d10, d4[1]
    vmlal.s16 q13, d11, d4[1]
    vmlal.s16 q14, d10, d6[1]
    vmlal.s16 q15, d11, d6[1]

    bls 2f

    // c2
    vld1.8 d9, [r7]!
    vmovl.s8 q4, d9
    vmlal.s16 q8, d8, d0[2]
    vmlal.s16 q9, d9, d0[2]
    vmlal.s16 q10, d8, d2[2]
    vmlal.s16 q11, d9, d2[2]
    vmlal.s16 q12, d8, d4[2]
    vmlal.s16 q13, d9, d4[2]
    vmlal.s16 q14, d8, d6[2]
    vmlal.s16 q15, d9, d6[2]

    cmp r2, #-32
    blo 2f

    // c3
    vld1.8 d11, [r7]!
    vmovl.s8 q5, d11
    vmlal.s16 q8, d10, d0[3]
    vmlal.s16 q9, d11, d0[3]
    vmlal.s16 q10, d10, d2[3]
    vmlal.s16 q11, d11, d2[3]
    vmlal.s16 q12, d10, d4[3]
    vmlal.s16 q13, d11, d4[3]
    vmlal.s16 q14, d10, d6[3]
    vmlal.s16 q15, d11, d6[3]

    bls 2f

    // c4
    vld1.8 d9, [r7]!
    vmovl.s8 q4, d9
    vmlal.s16 q8, d8, d1[0]
    vmlal.s16 q9, d9, d1[0]
    vmlal.s16 q10, d8, d3[0]
    vmlal.s16 q11, d9, d3[0]
    vmlal.s16 q12, d8, d5[0]
    vmlal.s16 q13, d9, d5[0]
    vmlal.s16 q14, d8, d7[0]
    vmlal.s16 q15, d9, d7[0]

    cmp r2, #-16
    blo 2f

    // c5
    vld1.8 d11, [r7]!
    vmovl.s8 q5, d11
    vmlal.s16 q8, d10, d1[1]
    vmlal.s16 q9, d11, d1[1]
    vmlal.s16 q10, d10, d3[1]
    vmlal.s16 q11, d11, d3[1]
    vmlal.s16 q12, d10, d5[1]
    vmlal.s16 q13, d11, d5[1]
    vmlal.s16 q14, d10, d7[1]
    vmlal.s16 q15, d11, d7[1]

    bls 2f

    // c6
    vld1.8 d9, [r7]!
    vmovl.s8 q4, d9
    vmlal.s16 q8, d8, d1[2]
    vmlal.s16 q9, d9, d1[2]
    vmlal.s16 q10, d8, d3[2]
    vmlal.s16 q11, d9, d3[2]
    vmlal.s16 q12, d8, d5[2]
    vmlal.s16 q13, d9, d5[2]
    vmlal.s16 q14, d8, d7[2]
    vmlal.s16 q15, d9, d7[2]

2:

    vld1.32 {d12,d13}, [r10]!    // float scales c0, c1, c2, c3
    vmov.i32 q7, #0
    cmp r1, #4
    ble 22f
    vld1.32 {d14,d15}, [r10]     // float scales c4, c5, c6, c7

22:
    cmp r11, #-1                 // relu (conv - relu - add, relu == -1)
    bne 23f
    vmov.i32 q0,  #0
    vmax.s32 q8,  q0
    vmax.s32 q9,  q0
    vmax.s32 q10, q0
    vmax.s32 q11, q0
    vmax.s32 q12, q0
    vmax.s32 q13, q0
    vmax.s32 q14, q0
    vmax.s32 q15, q0
23:
    vcvt.f32.s32 q8, q8          // result from int32 to float
    vcvt.f32.s32 q9, q9
    vcvt.f32.s32 q10, q10
    vcvt.f32.s32 q11, q11
    vcvt.f32.s32 q12, q12
    vcvt.f32.s32 q13, q13
    vcvt.f32.s32 q14, q14
    vcvt.f32.s32 q15, q15

    ldr r6, [sp, #124]           // add_input

    vmul.f32 q8, q8, q6          // result = result * scale
    vmul.f32 q9, q9, q7
    vmul.f32 q10, q10, q6
    vmul.f32 q11, q11, q7

    cmp r6, #0                   // if add_input_ptr == 0, skip
    beq 25f
    add r7, r6, r9
    cmp r0, #2
    movlo r7, r6

    vld1.s32 d0, [r6]
    vld1.s32 d2, [r7]
    vmovl.s8 q0,d0
    vmovl.s8 q1,d2
    vmovl.s16 q2,d0
    vmovl.s16 q3,d1
    vmovl.s16 q4,d2
    vmovl.s16 q5,d3

    ldr r10, [sp, #128]          // add_scale
    vld1.32 {d0,d1}, [r10]!
    vmov.i32 q1, #0
    cmp r1, #4
    ble 24f
    vld1.32 {d2,d3}, [r10]

24:
    vcvt.f32.s32 q2, q2
    vcvt.f32.s32 q3, q3
    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5

    vmla.f32 q8,  q2, q0         // result += add_input * add_scale
    vmla.f32 q9,  q3, q1
    vmla.f32 q10, q4, q0
    vmla.f32 q11, q5, q1

25:
    vmul.f32 q12, q12, q6        // result = result * scale
    vmul.f32 q13, q13, q7
    vmul.f32 q14, q14, q6
    vmul.f32 q15, q15, q7

    cmp r6, #0                   // if add_input_ptr == 0, skip
    beq 26f
    cmp r0, #2
    add r3, r7, r9
    movls r3, r7
    cmp r0, #4
    add r4, r3, r9
    movne r4, r3

    vld1.s32 d12, [r3]
    vld1.s32 d14, [r4]
    vmovl.s8 q6,d12
    vmovl.s8 q7,d14
    vmovl.s16 q2,d12
    vmovl.s16 q3,d13
    vmovl.s16 q4,d14
    vmovl.s16 q5,d15

    vcvt.f32.s32 q2, q2
    vcvt.f32.s32 q3, q3
    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5

    vmla.f32 q12, q2, q0        // result += add_input * add_scale
    vmla.f32 q13, q3, q1
    vmla.f32 q14, q4, q0
    vmla.f32 q15, q5, q1

26:
    // f32 --> s32 --> s8
    // val + (val >= 0.f ? 0.5f : -0.5f)
    vmov.f32 q0, #0.5
    vmov.f32 q1, #-0.5

    vcge.f32 q2, q8,  #0
    vcge.f32 q3, q9,  #0
    vcge.f32 q4, q10, #0
    vcge.f32 q5, q11, #0
    vcge.f32 q6, q12, #0
    vcge.f32 q7, q13, #0
    vbsl.f32 q2, q0, q1
    vbsl.f32 q3, q0, q1
    vbsl.f32 q4, q0, q1
    vbsl.f32 q5, q0, q1
    vbsl.f32 q6, q0, q1
    vbsl.f32 q7, q0, q1
    vadd.f32 q8, q8, q2
    vadd.f32 q9, q9, q3
    vcge.f32 q2, q14, #0
    vcge.f32 q3, q15, #0
    vadd.f32 q10, q10, q4
    vadd.f32 q11, q11, q5
    vadd.f32 q12, q12, q6
    vadd.f32 q13, q13, q7
    vbsl.f32 q2, q0, q1
    vbsl.f32 q3, q0, q1
    vadd.f32 q14, q14, q2
    vadd.f32 q15, q15, q3

    vcvt.s32.f32 q8, q8
    vcvt.s32.f32 q9, q9
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

    vqmovn.s16 d16, q8
    vqmovn.s16 d18, q9
    vqmovn.s16 d20, q10
    vqmovn.s16 d22, q11

    cmp r11, #1                // relu (conv add relu, relu == 1 or relu6 == 2)
    blt 3f
    vmov.i32 q0, #0
    vmax.s8 d16, d0
    vmax.s8 d18, d0
    vmax.s8 d20, d0
    vmax.s8 d22, d0

    cmp r11, #2                // relu6
    bne 3f
    ldr r11, [sp, #132]        // relu6_max
    vld1.s32 d0, [r11]
    vmin.s8 d16, d0
    vmin.s8 d18, d0
    vmin.s8 d20, d0
    vmin.s8 d22, d0

3:
    add r3, r8, r9
    cmp r0, #2
    movlo r3, r8

    add r4, r3, r9
    movls r4, r3

    cmp r0, #4
    add r5, r4, r9
    movne r5, r4

    cmp r1, #8
    bne 4f

    vst1.8 d16, [r8]
    vst1.8 d18, [r3]
    vst1.8 d20, [r4]
    vst1.8 d22, [r5]

4:
    cmp r1, #4
    blo 5f

    vst1.32 d16[0], [r8]!
    vst1.32 d18[0], [r3]!
    vst1.32 d20[0], [r4]!
    vst1.32 d22[0], [r5]!

    sub r1, r1, #4
    vext.8 d16, d16, #4
    vext.8 d18, d18, #4
    vext.8 d20, d20, #4
    vext.8 d22, d22, #4

5:
    cmp r1, #2
    blo 6f

    vst1.16 d16[0], [r8]!
    vst1.16 d18[0], [r3]!
    vst1.16 d20[0], [r4]!
    vst1.16 d22[0], [r5]!

    sub r1, r1, #2
    vext.8 d16, d16, #2
    vext.8 d18, d18, #2
    vext.8 d20, d20, #2
    vext.8 d22, d22, #2

6:
    cmp r1, #1
    blo 7f

    vst1.8 d16[0], [r8]!
    vst1.8 d18[0], [r3]!
    vst1.8 d20[0], [r4]!
    vst1.8 d22[0], [r5]!

7:

vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif
