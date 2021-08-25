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
#ifdef __aarch64__

#include "tnn/device/arm/acc/compute/asm_func_name.S"

.text
.align 5

.macro TRANSPOSE_2X16_S8 r0 r1
// r0: w0c0,w0c1,w0c2,w0c3,w0c4,w0c5,w0c6,w0c7 | w2c0,w2c1,w2c2,w2c3,w2c4,w2c5,w2c6,w2c7
// r1: w1c0,w1c1,w1c2,w1c3,w1c4,w1c5,w1c6,w1c7 | 0
    // w0c0,w1c0,w0c1,w1c1,w0c2,w1c2,w0c3,w1c3 | w0c4,w1c4,w0c5,w1c5,w0c6,w1c6,w0c7,w1c7
    zip1 v24.16b, \r0, \r1
    // w2c0,0,   w2c1,0,   w2c2,0,   w2c3,0,   | w2c4,0,   w2c5,0,   w2c6,0,   w2c7,0
    zip2 v25.16b, \r0, \r1
    // w0c0,w1c0,w2c0,0,w0c1,w1c1,w2c1,0,w0c2,w1c2,w2c2,0,w0c3,w1c3,w2c3,0
    zip1 v28.8h, v24.8h, v25.8h
    // w0c4,w1c4,w2c4,0,w0c5,w1c5,w2c5,0,w0c6,w1c6,w2c6,0,w0c7,w1c7,w2c7,0
    zip2 v29.8h, v24.8h, v25.8h
.endm

.macro TRANSPOSE_2X8_S8 r0 r1
// r0: w0c0,w0c1,w0c2,w0c3 | w2c0,w2c1,w2c2,w2c3
// r1: w1c0,w1c1,w1c2,w1c3 | 0
    // w0c0,w1c0,w0c1,w1c1,w0c2,w1c2,w0c3,w1c3
    zip1 v24.8b, \r0, \r1
    // w2c0,0,   w2c1,0,   w2c2,0,   w2c3,0
    zip2 v25.8b, \r0, \r1
    // w0c0,w1c0,w2c0,0,w0c1,w1c1,w2c1,0,w0c2,w1c2,w2c2,0,w0c3,w1c3,w2c3,0
    zip1 v28.4h, v24.4h, v25.4h
    zip2 v29.4h, v24.4h, v25.4h
    ins v28.d[1], v29.d[0]
.endm

asm_function ConvDw3x3S2Int8SdotSlideW
//void ConvDw3x3S2Int8SdotSlideW(int8_t *dst_z,
//                        int8_t **src,
//                        const int8_t* weight_z,
//                        const int32_t* bias_z,
//                        const float* scale_z,
//                        long dc,
//                        long dst_depth,
//                        long width)
//x0(dst_z),
//x1(int8_t** src),
//x2(weight_z),
//x3(bias_z),
//x4(scale_z),
//x5(dc),
//x6(dst_depth),
//x7(width)

cmp x7, #0
ble End

// weight
// c0 k0k1k2-, c1 k0k1k2-, c2 k0k1k2-, c3 k0k1k2-
ldr q0, [x2]
// c4 k0k1k2-, c5 k0k1k2-, c6 k0k1k2-, c7 k0k1k2-
ldr q1, [x2, #16]
ldr q2, [x2, #32]
ldr q3, [x2, #48]
ldr q4, [x2, #64]
ldr q5, [x2, #80]

// bias
ldr q6, [x3]
ldr q7, [x3, #16]

// scale
ldr q30, [x4]
ldr q31, [x4, #16]

ldr x9,  [x1]
ldr x10, [x1, #8]
ldr x11, [x1, #16]
add x9, x9, x5      // h0 ptr += dc
add x10, x10, x5    // h1 ptr += dc
add x11, x11, x5    // h2 ptr += dc

eor v17.16b, v17.16b, v17.16b

ld1 {v16.d}[0], [x9], x6
ld1 {v17.d}[0], [x10], x6
ld1 {v16.d}[1], [x11], x6
mov v18.16b, v6.16b
mov v19.16b, v7.16b
TRANSPOSE_2X16_S8 v16.16b, v17.16b
.word 0x4e809792 // sdot v18.4s, v28.16b, v0.16b
.word 0x4e8197b3 // sdot v19.4s, v29.16b, v1.16b

ld1 {v16.d}[0], [x9], x6
ld1 {v17.d}[0], [x10], x6
ld1 {v16.d}[1], [x11], x6
TRANSPOSE_2X16_S8 v16.16b, v17.16b
.word 0x4e829792 // sdot v18.4s, v28.16b, v2.16b
.word 0x4e8397b3 // sdot v19.4s, v29.16b, v3.16b

LoopDw:
    ld1 {v16.d}[0], [x9], x6
    ld1 {v17.d}[0], [x10], x6
    ld1 {v16.d}[1], [x11], x6
    mov v20.16b, v6.16b
    mov v21.16b, v7.16b
    prfm pldl1keep, [x9]
    prfm pldl1keep, [x10]
    prfm pldl1keep, [x11]
    TRANSPOSE_2X16_S8 v16.16b, v17.16b
    .word 0x4e849792 // sdot v18.4s, v28.16b, v4.16b
    .word 0x4e8597b3 // sdot v19.4s, v29.16b, v5.16b
    .word 0x4e809794 // sdot v20.4s, v28.16b, v0.16b
    .word 0x4e8197b5 // sdot v21.4s, v29.16b, v1.16b

    ld1 {v16.d}[0], [x9], x6
    ld1 {v17.d}[0], [x10], x6
    ld1 {v16.d}[1], [x11], x6
    prfm pldl1keep, [x9]
    prfm pldl1keep, [x10]
    prfm pldl1keep, [x11]
    TRANSPOSE_2X16_S8 v16.16b, v17.16b
    .word 0x4e829794 // sdot v20.4s, v28.16b, v2.16b
    .word 0x4e8397b5 // sdot v21.4s, v29.16b, v3.16b

    subs x7, x7, #1

    scvtf v26.4s, v18.4s
    scvtf v27.4s, v19.4s
    mov v18.16b, v20.16b
    mov v19.16b, v21.16b
    fmul v26.4s, v26.4s, v30.4s  // result *= scale
    fmul v27.4s, v27.4s, v31.4s
    fcvtas v26.4s, v26.4s
    fcvtas v27.4s, v27.4s
    sqxtn  v26.4h, v26.4s
    sqxtn2 v26.8h, v27.4s
    sqxtn v26.8b, v26.8h
    st1 {v26.8b}, [x0], x6

    bne LoopDw

End:

ret

asm_function ConvDw3x3S2Int8SdotSlideWLeftC4
//void ConvDw3x3S2Int8SdotSlideWLeftC4(int8_t *dst_z,
//                        int8_t **src,
//                        const int8_t* weight_z,
//                        const int32_t* bias_z,
//                        const float* scale_z,
//                        long dc,
//                        long dst_depth,
//                        long width)
//x0(dst_z),
//x1(int8_t** src),
//x2(weight_z),
//x3(bias_z),
//x4(scale_z),
//x5(dc),
//x6(dst_depth),
//x7(width)

cmp x7, #0
ble C4End

// weight
// c0 k0k1k2-, c1 k0k1k2-, c2 k0k1k2-, c3 k0k1k2-
ldr q0, [x2]
ldr q1, [x2, #16]
ldr q2, [x2, #32]

// bias
ldr q3, [x3]

// scale
ldr q4, [x4]

ldr x9,  [x1]
ldr x10, [x1, #8]
ldr x11, [x1, #16]
add x9, x9, x5      // h0 ptr += dc
add x10, x10, x5    // h1 ptr += dc
add x11, x11, x5    // h2 ptr += dc

eor v17.16b, v17.16b, v17.16b

ld1 {v16.s}[0], [x9], x6
ld1 {v17.s}[0], [x10], x6
ld1 {v16.s}[1], [x11], x6
mov v18.16b, v3.16b
TRANSPOSE_2X8_S8 v16.8b, v17.8b
.word 0x4e809792 // sdot v18.4s, v28.16b, v0.16b

ld1 {v16.s}[0], [x9], x6
ld1 {v17.s}[0], [x10], x6
ld1 {v16.s}[1], [x11], x6
TRANSPOSE_2X8_S8 v16.8b, v17.8b
.word 0x4e819792 // sdot v18.4s, v28.16b, v1.16b

C4LoopDw:
    ld1 {v16.s}[0], [x9], x6
    ld1 {v17.s}[0], [x10], x6
    ld1 {v16.s}[1], [x11], x6
    mov v19.16b, v3.16b
    prfm pldl1keep, [x9]
    prfm pldl1keep, [x10]
    prfm pldl1keep, [x11]
    TRANSPOSE_2X8_S8 v16.8b, v17.8b
    .word 0x4e829792 // sdot v18.4s, v28.16b, v2.16b
    .word 0x4e809793 // sdot v19.4s, v28.16b, v0.16b

    ld1 {v16.s}[0], [x9], x6
    ld1 {v17.s}[0], [x10], x6
    ld1 {v16.s}[1], [x11], x6
    prfm pldl1keep, [x9]
    prfm pldl1keep, [x10]
    prfm pldl1keep, [x11]
    TRANSPOSE_2X8_S8 v16.8b, v17.8b
    .word 0x4e819793 // sdot v19.4s, v28.16b, v1.16b

    subs x7, x7, #1

    scvtf v21.4s, v18.4s
    mov v18.16b, v19.16b
    fmul v21.4s, v21.4s, v4.4s
    fcvtas v21.4s, v21.4s
    sqxtn v21.4h, v21.4s
    sqxtn v21.8b, v21.8h
    st1 {v21.s}[0], [x0], x6

    bne C4LoopDw

C4End:

ret

#endif
#endif
