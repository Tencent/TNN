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
asm_function GemmInt8SdotUnit8x4
//void GemmInt8SdotUnit8x4(int8_t* dst, const int8_t* src, const int8_t* weight,
//                         long src_depth, long dst_depth, long hw, 
//                         const int32_t* bias, const float* scale,
//                         long relu, const int8_t* add_input, 
//                         const float* add_scale, const int8_t* relu6_max)
//x0(dst),
//x1(src),
//x2(weight),
//x3(src_depth),
//x4(dst_depth),
//x5(hw),
//x6(bias),
//x7(scale)
//from stack(relu)      [sp, #0]
//from stack(add_input) [sp, #8]
//from stack(add_scale) [sp, #16]
//from stack(relu6_max) [sp, #24]

sub sp, sp, #192
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
stp x19, x20, [sp], #16
stp x21, x22, [sp], #16
stp x23, x24, [sp], #16
stp x25, x26, [sp], #16

// relu
ldr x23, [sp, #0]
// zero vector
eor v31.16b, v31.16b, v31.16b
// add_input
ldr x25, [sp, #8]
// add_scale
ldr x26, [sp, #16]
// relu6_max
ldr x24, [sp, #24]

// src_ptr x1
// hw counter x5
// dst_ptr x0

LoopHW8:
    // if hw counter <= 7, skip
    cmp x5, #7
    ble LoopHW1

    // src_ptr 0 ~ 7
    mov x10, x1
    add x11, x1, x3
    add x12, x1, x3, lsl#1
    add x14, x1, x3, lsl#2
    add x13, x11, x3, lsl#1
    add x15, x11, x3, lsl#2
    add x19, x12, x3, lsl#2
    add x20, x13, x3, lsl#2

    // load bias 32bit, accumulator 8 reg
    ld1 {v16.4s}, [x6]
    mov v17.16b, v16.16b
    mov v18.16b, v16.16b
    mov v19.16b, v16.16b
    mov v20.16b, v16.16b
    mov v21.16b, v16.16b
    mov v22.16b, v16.16b
    mov v23.16b, v16.16b

    // src_depth counter
    mov x21, x3

    // weight_ptr
    mov x22, x2

    cmp x21, #15
    ble LoopCrr8

    ldr q8, [x22]
    ldr q9, [x22, #16]
    ldr q10, [x22, #32]
    ldr q11, [x22, #48]

    ld1 {v0.16b}, [x10], #16
    ld1 {v1.16b}, [x11], #16
    ld1 {v2.16b}, [x12], #16
    ld1 {v3.16b}, [x13], #16
    ld1 {v4.16b}, [x14], #16
    ld1 {v5.16b}, [x15], #16
    ld1 {v6.16b}, [x19], #16
    ld1 {v7.16b}, [x20], #16

    .word 0x4f80e110 // sdot v16.4s, v8.16b,  v0.4b[0]
    .word 0x4f81e111 // sdot v17.4s, v8.16b,  v1.4b[0]
    .word 0x4f82e112 // sdot v18.4s, v8.16b,  v2.4b[0]
    .word 0x4f83e113 // sdot v19.4s, v8.16b,  v3.4b[0]
    .word 0x4f84e114 // sdot v20.4s, v8.16b,  v4.4b[0]
    .word 0x4f85e115 // sdot v21.4s, v8.16b,  v5.4b[0]
    .word 0x4f86e116 // sdot v22.4s, v8.16b,  v6.4b[0]
    .word 0x4f87e117 // sdot v23.4s, v8.16b,  v7.4b[0]

    sub x21, x21, #16

    LoopCrr16:
        cmp x21, #15
        ble LoopCrr16End

        add x22, x22, #64

        .word 0x4fa0e130 // sdot v16.4s, v9.16b,  v0.4b[1]
        .word 0x4fa1e131 // sdot v17.4s, v9.16b,  v1.4b[1]
        .word 0x4fa2e132 // sdot v18.4s, v9.16b,  v2.4b[1]
        .word 0x4fa3e133 // sdot v19.4s, v9.16b,  v3.4b[1]
        .word 0x4fa4e134 // sdot v20.4s, v9.16b,  v4.4b[1]
        .word 0x4fa5e135 // sdot v21.4s, v9.16b,  v5.4b[1]
        .word 0x4fa6e136 // sdot v22.4s, v9.16b,  v6.4b[1]
        .word 0x4fa7e137 // sdot v23.4s, v9.16b,  v7.4b[1]

        ldr q8, [x22]
        ldr q9, [x22, #16]

        .word 0x4f80e950 // sdot v16.4s, v10.16b, v0.4b[2]
        .word 0x4f81e951 // sdot v17.4s, v10.16b, v1.4b[2]
        .word 0x4f82e952 // sdot v18.4s, v10.16b, v2.4b[2]
        .word 0x4f83e953 // sdot v19.4s, v10.16b, v3.4b[2]
        .word 0x4f84e954 // sdot v20.4s, v10.16b, v4.4b[2]
        .word 0x4f85e955 // sdot v21.4s, v10.16b, v5.4b[2]
        .word 0x4f86e956 // sdot v22.4s, v10.16b, v6.4b[2]
        .word 0x4f87e957 // sdot v23.4s, v10.16b, v7.4b[2]

        ldr q10, [x22, #32]

        .word 0x4fa0e970 // sdot v16.4s, v11.16b, v0.4b[3]
        ld1 {v0.16b}, [x10], #16
        .word 0x4fa1e971 // sdot v17.4s, v11.16b, v1.4b[3]
        ld1 {v1.16b}, [x11], #16
        .word 0x4fa2e972 // sdot v18.4s, v11.16b, v2.4b[3]
        ld1 {v2.16b}, [x12], #16
        .word 0x4fa3e973 // sdot v19.4s, v11.16b, v3.4b[3]
        ld1 {v3.16b}, [x13], #16
        .word 0x4fa4e974 // sdot v20.4s, v11.16b, v4.4b[3]
        ld1 {v4.16b}, [x14], #16
        .word 0x4fa5e975 // sdot v21.4s, v11.16b, v5.4b[3]
        ld1 {v5.16b}, [x15], #16
        .word 0x4fa6e976 // sdot v22.4s, v11.16b, v6.4b[3]
        ld1 {v6.16b}, [x19], #16
        .word 0x4fa7e977 // sdot v23.4s, v11.16b, v7.4b[3]
        ld1 {v7.16b}, [x20], #16

        ldr q11, [x22, #48]
        sub x21, x21, #16

        .word 0x4f80e110 // sdot v16.4s, v8.16b,  v0.4b[0]
        .word 0x4f81e111 // sdot v17.4s, v8.16b,  v1.4b[0]
        .word 0x4f82e112 // sdot v18.4s, v8.16b,  v2.4b[0]
        .word 0x4f83e113 // sdot v19.4s, v8.16b,  v3.4b[0]
        .word 0x4f84e114 // sdot v20.4s, v8.16b,  v4.4b[0]
        .word 0x4f85e115 // sdot v21.4s, v8.16b,  v5.4b[0]
        .word 0x4f86e116 // sdot v22.4s, v8.16b,  v6.4b[0]
        .word 0x4f87e117 // sdot v23.4s, v8.16b,  v7.4b[0]

        b LoopCrr16

    LoopCrr16End:

        add x22, x22, #64
        .word 0x4fa0e130 // sdot v16.4s, v9.16b,  v0.4b[1]
        .word 0x4fa1e131 // sdot v17.4s, v9.16b,  v1.4b[1]
        .word 0x4fa2e132 // sdot v18.4s, v9.16b,  v2.4b[1]
        .word 0x4fa3e133 // sdot v19.4s, v9.16b,  v3.4b[1]
        .word 0x4fa4e134 // sdot v20.4s, v9.16b,  v4.4b[1]
        .word 0x4fa5e135 // sdot v21.4s, v9.16b,  v5.4b[1]
        .word 0x4fa6e136 // sdot v22.4s, v9.16b,  v6.4b[1]
        .word 0x4fa7e137 // sdot v23.4s, v9.16b,  v7.4b[1]
        .word 0x4f80e950 // sdot v16.4s, v10.16b, v0.4b[2]
        .word 0x4f81e951 // sdot v17.4s, v10.16b, v1.4b[2]
        .word 0x4f82e952 // sdot v18.4s, v10.16b, v2.4b[2]
        .word 0x4f83e953 // sdot v19.4s, v10.16b, v3.4b[2]
        .word 0x4f84e954 // sdot v20.4s, v10.16b, v4.4b[2]
        .word 0x4f85e955 // sdot v21.4s, v10.16b, v5.4b[2]
        .word 0x4f86e956 // sdot v22.4s, v10.16b, v6.4b[2]
        .word 0x4f87e957 // sdot v23.4s, v10.16b, v7.4b[2]
        .word 0x4fa0e970 // sdot v16.4s, v11.16b, v0.4b[3]
        .word 0x4fa1e971 // sdot v17.4s, v11.16b, v1.4b[3]
        .word 0x4fa2e972 // sdot v18.4s, v11.16b, v2.4b[3]
        .word 0x4fa3e973 // sdot v19.4s, v11.16b, v3.4b[3]
        .word 0x4fa4e974 // sdot v20.4s, v11.16b, v4.4b[3]
        .word 0x4fa5e975 // sdot v21.4s, v11.16b, v5.4b[3]
        .word 0x4fa6e976 // sdot v22.4s, v11.16b, v6.4b[3]
        .word 0x4fa7e977 // sdot v23.4s, v11.16b, v7.4b[3]

    LoopCrr8:
        cmp x21, #7
        ble LoopCrr4

        ld1 {v8.16b, v9.16b}, [x22], #32

        ld1 {v0.8b}, [x10], #8
        ld1 {v1.8b}, [x11], #8
        ld1 {v2.8b}, [x12], #8
        ld1 {v3.8b}, [x13], #8
        ld1 {v4.8b}, [x14], #8
        ld1 {v5.8b}, [x15], #8
        ld1 {v6.8b}, [x19], #8
        ld1 {v7.8b}, [x20], #8

        .word 0x4f80e110 // sdot v16.4s, v8.16b, v0.4b[0]
        .word 0x4f81e111 // sdot v17.4s, v8.16b, v1.4b[0]
        .word 0x4f82e112 // sdot v18.4s, v8.16b, v2.4b[0]
        .word 0x4f83e113 // sdot v19.4s, v8.16b, v3.4b[0]
        .word 0x4f84e114 // sdot v20.4s, v8.16b, v4.4b[0]
        .word 0x4f85e115 // sdot v21.4s, v8.16b, v5.4b[0]
        .word 0x4f86e116 // sdot v22.4s, v8.16b, v6.4b[0]
        .word 0x4f87e117 // sdot v23.4s, v8.16b, v7.4b[0]

        sub x21, x21, #8

        .word 0x4fa0e130 // sdot v16.4s, v9.16b, v0.4b[1]
        .word 0x4fa1e131 // sdot v17.4s, v9.16b, v1.4b[1]
        .word 0x4fa2e132 // sdot v18.4s, v9.16b, v2.4b[1]
        .word 0x4fa3e133 // sdot v19.4s, v9.16b, v3.4b[1]
        .word 0x4fa4e134 // sdot v20.4s, v9.16b, v4.4b[1]
        .word 0x4fa5e135 // sdot v21.4s, v9.16b, v5.4b[1]
        .word 0x4fa6e136 // sdot v22.4s, v9.16b, v6.4b[1]
        .word 0x4fa7e137 // sdot v23.4s, v9.16b, v7.4b[1]

        b LoopCrr8
    
    LoopCrr4:
        cmp x21, #3
        ble LoopEnd

        ld1 {v8.16b}, [x22], #16
        ld1 {v0.s}[0], [x10], #4
        ld1 {v1.s}[0], [x11], #4
        ld1 {v2.s}[0], [x12], #4
        ld1 {v3.s}[0], [x13], #4
        ld1 {v4.s}[0], [x14], #4
        ld1 {v5.s}[0], [x15], #4
        ld1 {v6.s}[0], [x19], #4
        ld1 {v7.s}[0], [x20], #4

        sub x21, x21, #4
        .word 0x4f80e110 // sdot v16.4s, v8.16b,  v0.4b[0]
        .word 0x4f81e111 // sdot v17.4s, v8.16b,  v1.4b[0]
        .word 0x4f82e112 // sdot v18.4s, v8.16b,  v2.4b[0]
        .word 0x4f83e113 // sdot v19.4s, v8.16b,  v3.4b[0]
        .word 0x4f84e114 // sdot v20.4s, v8.16b,  v4.4b[0]
        .word 0x4f85e115 // sdot v21.4s, v8.16b,  v5.4b[0]
        .word 0x4f86e116 // sdot v22.4s, v8.16b,  v6.4b[0]
        .word 0x4f87e117 // sdot v23.4s, v8.16b,  v7.4b[0]

        b LoopCrr4

LoopEnd:
    // hw counter -= 8
    sub x5, x5, #8
    // src_ptr += 8 * src_depth
    add x1, x1, x3, lsl#3

    // scale oc0 ~ oc7
    ldr q1, [x7]

ConvReluAdd:
    cmp x23, #-1  // if relu == -1, Conv-Relu-Add
    bne MulScale

    smax v16.4s, v16.4s, v31.4s
    smax v17.4s, v17.4s, v31.4s
    smax v18.4s, v18.4s, v31.4s
    smax v19.4s, v19.4s, v31.4s
    smax v20.4s, v20.4s, v31.4s
    smax v21.4s, v21.4s, v31.4s
    smax v22.4s, v22.4s, v31.4s
    smax v23.4s, v23.4s, v31.4s
MulScale:
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s

    fmul v16.4s, v16.4s, v1.4s
    fmul v17.4s, v17.4s, v1.4s
    fmul v18.4s, v18.4s, v1.4s
    fmul v19.4s, v19.4s, v1.4s
    fmul v20.4s, v20.4s, v1.4s
    fmul v21.4s, v21.4s, v1.4s
    fmul v22.4s, v22.4s, v1.4s
    fmul v23.4s, v23.4s, v1.4s

    cbz x25, ConvAddPost  // if add_input_ptr == 0, skip

AddInputScale:

    // add_input_ptr 0 ~ 7
    add x11, x25, x4
    add x12, x25, x4, lsl#1
    add x14, x25, x4, lsl#2
    add x13, x11, x4, lsl#1
    add x15, x11, x4, lsl#2
    add x19, x12, x4, lsl#2
    add x20, x13, x4, lsl#2
    ld1 {v0.s}[0], [x25]
    ld1 {v1.s}[0], [x11]
    ld1 {v2.s}[0], [x12]
    ld1 {v3.s}[0], [x13]
    ld1 {v4.s}[0], [x14]
    ld1 {v5.s}[0], [x15]
    ld1 {v6.s}[0], [x19]
    ld1 {v7.s}[0], [x20]
    // add_scale
    ldr q8, [x26]

    // convert add_input int8 to fp32
    sxtl v0.8h, v0.8b
    sxtl v1.8h, v1.8b
    sxtl v2.8h, v2.8b
    sxtl v3.8h, v3.8b
    sxtl v4.8h, v4.8b
    sxtl v5.8h, v5.8b
    sxtl v6.8h, v6.8b
    sxtl v7.8h, v7.8b
    sxtl v0.4s, v0.4h
    sxtl v1.4s, v1.4h
    sxtl v2.4s, v2.4h
    sxtl v3.4s, v3.4h
    sxtl v4.4s, v4.4h
    sxtl v5.4s, v5.4h
    sxtl v6.4s, v6.4h
    sxtl v7.4s, v7.4h
    scvtf v0.4s, v0.4s
    scvtf v1.4s, v1.4s
    scvtf v2.4s, v2.4s
    scvtf v3.4s, v3.4s
    scvtf v4.4s, v4.4s
    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    scvtf v7.4s, v7.4s

    fmla v16.4s, v0.4s, v8.4s
    fmla v17.4s, v1.4s, v8.4s
    fmla v18.4s, v2.4s, v8.4s
    fmla v19.4s, v3.4s, v8.4s
    fmla v20.4s, v4.4s, v8.4s
    fmla v21.4s, v5.4s, v8.4s
    fmla v22.4s, v6.4s, v8.4s
    fmla v23.4s, v7.4s, v8.4s

    // add_input_ptr += 8 * dst_depth
    add x25, x25, x4, lsl#3

ConvAddPost:
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s
    fcvtas v22.4s, v22.4s
    fcvtas v23.4s, v23.4s

    sqxtn  v16.4h, v16.4s
    sqxtn  v18.4h, v18.4s
    sqxtn  v20.4h, v20.4s
    sqxtn  v22.4h, v22.4s
    sqxtn2 v16.8h, v17.4s
    sqxtn2 v18.8h, v19.4s
    sqxtn2 v20.8h, v21.4s
    sqxtn2 v22.8h, v23.4s

    sqxtn v16.8b, v16.8h
    sqxtn v18.8b, v18.8h
    sqxtn v20.8b, v20.8h
    sqxtn v22.8b, v22.8h

    cmp x23, #1  // if relu != 1 or 2, Conv-Add-Relu or Relu6, skip
    blt ConvAddPostEnd
    smax v16.8b, v16.8b, v31.8b
    smax v18.8b, v18.8b, v31.8b
    smax v20.8b, v20.8b, v31.8b
    smax v22.8b, v22.8b, v31.8b

    cmp x23, #2   // relu6
    bne ConvAddPostEnd
    ld1r {v0.2s}, [x24]
    smin v16.8b, v16.8b, v0.8b
    smin v18.8b, v18.8b, v0.8b
    smin v20.8b, v20.8b, v0.8b
    smin v22.8b, v22.8b, v0.8b

ConvAddPostEnd:

    // store to dst_ptr 0 ~ 7
    mov x10, x0
    add x11, x0,  x4
    add x12, x0,  x4, lsl#1
    add x14, x0,  x4, lsl#2
    add x13, x11, x4, lsl#1
    add x15, x11, x4, lsl#2
    add x19, x12, x4, lsl#2
    add x20, x13, x4, lsl#2

    st1 {v16.s}[0], [x10]
    st1 {v16.s}[1], [x11]
    st1 {v18.s}[0], [x12]
    st1 {v18.s}[1], [x13]
    st1 {v20.s}[0], [x14]
    st1 {v20.s}[1], [x15]
    st1 {v22.s}[0], [x19]
    st1 {v22.s}[1], [x20]

    // dst_ptr += 8 * dst_depth
    add x0, x0, x4, lsl#3

    b LoopHW8

LoopHW1:
    // if hw counter <= 0, skip
    cmp x5, #0
    ble LoopHW1End

    // src_ptr_0
    mov x10, x1

    // load bias 32bit, accumulator 2 reg
    ld1 {v16.4s}, [x6]

    // src_depth counter
    mov x21, x3

    // weight_ptr
    mov x22, x2

    HW1LoopCrr16:
        cmp x21, #15
        ble HW1LoopCrr8

        ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x22], #64
        ld1 {v0.16b}, [x10], #16
        sub x21, x21, #16
        .word 0x4f80e110 // sdot v16.4s, v8.16b,  v0.4b[0]
        .word 0x4fa0e130 // sdot v16.4s, v9.16b,  v0.4b[1]
        .word 0x4f80e950 // sdot v16.4s, v10.16b, v0.4b[2]
        .word 0x4fa0e970 // sdot v16.4s, v11.16b, v0.4b[3]
        b HW1LoopCrr16
    
    HW1LoopCrr8:
        cmp x21, #7
        ble HW1LoopCrr4

        ld1 {v8.16b, v9.16b}, [x22], #32
        ld1 {v0.8b}, [x10], #8
        sub x21, x21, #8
        .word 0x4f80e110 // sdot v16.4s, v8.16b, v0.4b[0]
        .word 0x4fa0e130 // sdot v16.4s, v9.16b, v0.4b[1]
        b HW1LoopCrr8

    HW1LoopCrr4:
        cmp x21, #3
        ble HW1LoopEnd

        ld1 {v8.16b}, [x22], #16
        ld1 {v0.s}[0], [x10], #4
        sub x21, x21, #4
        .word 0x4f80e110 // sdot v16.4s, v8.16b, v0.4b[0]
        b HW1LoopCrr4

HW1LoopEnd:
    // hw counter -= 1
    sub x5, x5, #1
    // src_ptr += 1 * src_depth
    add x1, x1, x3

    // scale oc0 ~ oc7
    ldr q1, [x7]

HW1ConvReluAdd:
    cmp x23, #-1  // if relu == -1, Conv-Relu-Add
    bne HW1MulScale

    smax v16.4s, v16.4s, v31.4s

HW1MulScale:
    scvtf v16.4s, v16.4s

    fmul v16.4s, v16.4s, v1.4s

    cbz x25, HW1ConvAddPost  // if add_input_ptr == 0, skip

HW1AddInputScale:

    ld1 {v0.s}[0], [x25]
    // add_scale
    ldr q1, [x26]
    // convert add_input int8 to fp32
    sxtl v0.8h, v0.8b
    sxtl v0.4s, v0.4h
    scvtf v0.4s, v0.4s
    fmla v16.4s, v0.4s, v1.4s

    // add_input_ptr += 1 * dst_depth
    add x25, x25, x4

HW1ConvAddPost:
    fcvtas v16.4s, v16.4s
    sqxtn v16.4h, v16.4s
    sqxtn v16.8b, v16.8h

    cmp x23, #1  // if relu != 1 or 2, Conv-Add-Relu or Relu6, skip
    blt HW1ConvAddPostEnd
    smax v16.8b, v16.8b, v31.8b

    cmp x23, #2   // relu6
    bne HW1ConvAddPostEnd
    ld1 {v0.s}[0], [x24]
    smin v16.8b, v16.8b, v0.8b

HW1ConvAddPostEnd:
    // store to dst_ptr
    st1 {v16.s}[0], [x0]
    // dst_ptr += 1 * dst_depth
    add x0, x0, x4

    b LoopHW1

LoopHW1End:

sub sp, sp, #192
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ldp x19, x20, [sp], #16
ldp x21, x22, [sp], #16
ldp x23, x24, [sp], #16
ldp x25, x26, [sp], #16

END:
ret

#endif
#endif
