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
asm_function GemmInt8SdotUnit8x8
//void GemmInt8SdotUnit8x8(int8_t* dst, const int8_t* src, const int8_t* weight,
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

// dz counter, dst_depth / 8 * 8
lsr x9, x4, #3
lsl x9, x9, #3

subs x9, x9, #8
blt END

sub sp, sp, #208
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
stp x19, x20, [sp], #16
stp x21, x22, [sp], #16
stp x23, x24, [sp], #16
stp x25, x26, [sp], #16
stp x27, x28, [sp], #16

// add_input
ldr x26, [sp, #8]
// add_scale
ldr x27, [sp, #16]

// for (long dz = 0; dz + 7 < dst_depth; dz += 8)
LoopDz8:

// src_ptr
mov x23, x1

// hw counter
mov x24, x5

// dst_ptr
mov x25, x0

// add_input_ptr
mov x28, x26

LoopHW8:
    // if hw counter <= 7, skip
    cmp x24, #7
    ble LoopHW1

    // src_ptr 0 ~ 7
    mov x10, x23
    add x11, x23, x3
    add x12, x23, x3, lsl#1
    add x14, x23, x3, lsl#2
    add x13, x11, x3, lsl#1
    add x15, x11, x3, lsl#2
    add x19, x12, x3, lsl#2
    add x20, x13, x3, lsl#2

    // load bias 32bit, accumulator 16 reg
    ld1 {v16.4s, v17.4s}, [x6]
    mov v18.16b, v16.16b
    mov v19.16b, v17.16b
    mov v20.16b, v16.16b
    mov v21.16b, v17.16b
    mov v22.16b, v16.16b
    mov v23.16b, v17.16b
    mov v24.16b, v16.16b
    mov v25.16b, v17.16b
    mov v26.16b, v16.16b
    mov v27.16b, v17.16b
    mov v28.16b, v16.16b
    mov v29.16b, v17.16b
    mov v30.16b, v16.16b
    mov v31.16b, v17.16b

    // src_depth counter
    mov x21, x3

    // weight_ptr
    mov x22, x2

    cmp x21, #15
    ble LoopCrr8

    ldr q8, [x22]
    ldr q9, [x22, #16]

    ld1 {v0.16b}, [x10], #16
    ld1 {v1.16b}, [x11], #16
    ld1 {v2.16b}, [x12], #16
    ld1 {v3.16b}, [x13], #16
    ld1 {v4.16b}, [x14], #16
    ld1 {v5.16b}, [x15], #16
    ld1 {v6.16b}, [x19], #16
    ld1 {v7.16b}, [x20], #16

    ldr q10, [x22, #32]
    ldr q11, [x22, #48]

    .word 0x4f80e110 // sdot v16.4s, v8.16b,  v0.4b[0]
    .word 0x4f80e131 // sdot v17.4s, v9.16b,  v0.4b[0]
    .word 0x4f81e112 // sdot v18.4s, v8.16b,  v1.4b[0]
    .word 0x4f81e133 // sdot v19.4s, v9.16b,  v1.4b[0]
    .word 0x4f82e114 // sdot v20.4s, v8.16b,  v2.4b[0]
    .word 0x4f82e135 // sdot v21.4s, v9.16b,  v2.4b[0]
    .word 0x4f83e116 // sdot v22.4s, v8.16b,  v3.4b[0]
    .word 0x4f83e137 // sdot v23.4s, v9.16b,  v3.4b[0]
    .word 0x4f84e118 // sdot v24.4s, v8.16b,  v4.4b[0]
    .word 0x4f84e139 // sdot v25.4s, v9.16b,  v4.4b[0]
    .word 0x4f85e11a // sdot v26.4s, v8.16b,  v5.4b[0]
    .word 0x4f85e13b // sdot v27.4s, v9.16b,  v5.4b[0]
    .word 0x4f86e11c // sdot v28.4s, v8.16b,  v6.4b[0]
    .word 0x4f86e13d // sdot v29.4s, v9.16b,  v6.4b[0]
    .word 0x4f87e11e // sdot v30.4s, v8.16b,  v7.4b[0]
    .word 0x4f87e13f // sdot v31.4s, v9.16b,  v7.4b[0]

    sub x21, x21, #16

    LoopCrr16:
        cmp x21, #15
        ble LoopCrr16End

        ldr q8, [x22, #64]
        ldr q9, [x22, #80]

        .word 0x4fa0e150 // sdot v16.4s, v10.16b, v0.4b[1]
        .word 0x4fa0e171 // sdot v17.4s, v11.16b, v0.4b[1]
        prfm pldl1keep, [x10, #64]
        .word 0x4fa1e152 // sdot v18.4s, v10.16b, v1.4b[1]
        .word 0x4fa1e173 // sdot v19.4s, v11.16b, v1.4b[1]
        prfm pldl1keep, [x11, #64]
        .word 0x4fa2e154 // sdot v20.4s, v10.16b, v2.4b[1]
        .word 0x4fa2e175 // sdot v21.4s, v11.16b, v2.4b[1]
        prfm pldl1keep, [x12, #64]
        .word 0x4fa3e156 // sdot v22.4s, v10.16b, v3.4b[1]
        .word 0x4fa3e177 // sdot v23.4s, v11.16b, v3.4b[1]
        prfm pldl1keep, [x13, #64]
        .word 0x4fa4e158 // sdot v24.4s, v10.16b, v4.4b[1]
        .word 0x4fa4e179 // sdot v25.4s, v11.16b, v4.4b[1]
        prfm pldl1keep, [x14, #64]
        .word 0x4fa5e15a // sdot v26.4s, v10.16b, v5.4b[1]
        .word 0x4fa5e17b // sdot v27.4s, v11.16b, v5.4b[1]
        prfm pldl1keep, [x15, #64]
        .word 0x4fa6e15c // sdot v28.4s, v10.16b, v6.4b[1]
        .word 0x4fa6e17d // sdot v29.4s, v11.16b, v6.4b[1]
        prfm pldl1keep, [x19, #64]
        .word 0x4fa7e15e // sdot v30.4s, v10.16b, v7.4b[1]
        .word 0x4fa7e17f // sdot v31.4s, v11.16b, v7.4b[1]
        prfm pldl1keep, [x20, #64]

        ldr q10, [x22, #96]
        ldr q11, [x22, #112]
        add x22, x22, #128

        .word 0x4f80e910 // sdot v16.4s, v8.16b,  v0.4b[2]
        .word 0x4f80e931 // sdot v17.4s, v9.16b,  v0.4b[2]
        .word 0x4f81e912 // sdot v18.4s, v8.16b,  v1.4b[2]
        .word 0x4f81e933 // sdot v19.4s, v9.16b,  v1.4b[2]
        .word 0x4f82e914 // sdot v20.4s, v8.16b,  v2.4b[2]
        .word 0x4f82e935 // sdot v21.4s, v9.16b,  v2.4b[2]
        .word 0x4f83e916 // sdot v22.4s, v8.16b,  v3.4b[2]
        .word 0x4f83e937 // sdot v23.4s, v9.16b,  v3.4b[2]
        .word 0x4f84e918 // sdot v24.4s, v8.16b,  v4.4b[2]
        .word 0x4f84e939 // sdot v25.4s, v9.16b,  v4.4b[2]
        .word 0x4f85e91a // sdot v26.4s, v8.16b,  v5.4b[2]
        .word 0x4f85e93b // sdot v27.4s, v9.16b,  v5.4b[2]
        .word 0x4f86e91c // sdot v28.4s, v8.16b,  v6.4b[2]
        .word 0x4f86e93d // sdot v29.4s, v9.16b,  v6.4b[2]
        .word 0x4f87e91e // sdot v30.4s, v8.16b,  v7.4b[2]
        .word 0x4f87e93f // sdot v31.4s, v9.16b,  v7.4b[2]

        ldr q8, [x22]
        ldr q9, [x22, #16]

        .word 0x4fa0e950 // sdot v16.4s, v10.16b, v0.4b[3]
        .word 0x4fa0e971 // sdot v17.4s, v11.16b, v0.4b[3]
        ld1 {v0.16b}, [x10], #16
        .word 0x4fa1e952 // sdot v18.4s, v10.16b, v1.4b[3]
        .word 0x4fa1e973 // sdot v19.4s, v11.16b, v1.4b[3]
        ld1 {v1.16b}, [x11], #16
        .word 0x4fa2e954 // sdot v20.4s, v10.16b, v2.4b[3]
        .word 0x4fa2e975 // sdot v21.4s, v11.16b, v2.4b[3]
        ld1 {v2.16b}, [x12], #16
        .word 0x4fa3e956 // sdot v22.4s, v10.16b, v3.4b[3]
        .word 0x4fa3e977 // sdot v23.4s, v11.16b, v3.4b[3]
        ld1 {v3.16b}, [x13], #16
        .word 0x4fa4e958 // sdot v24.4s, v10.16b, v4.4b[3]
        .word 0x4fa4e979 // sdot v25.4s, v11.16b, v4.4b[3]
        ld1 {v4.16b}, [x14], #16
        .word 0x4fa5e95a // sdot v26.4s, v10.16b, v5.4b[3]
        .word 0x4fa5e97b // sdot v27.4s, v11.16b, v5.4b[3]
        ld1 {v5.16b}, [x15], #16
        .word 0x4fa6e95c // sdot v28.4s, v10.16b, v6.4b[3]
        .word 0x4fa6e97d // sdot v29.4s, v11.16b, v6.4b[3]
        ld1 {v6.16b}, [x19], #16
        .word 0x4fa7e95e // sdot v30.4s, v10.16b, v7.4b[3]
        .word 0x4fa7e97f // sdot v31.4s, v11.16b, v7.4b[3]
        ld1 {v7.16b}, [x20], #16

        ldr q10, [x22, #32]
        ldr q11, [x22, #48]
        sub x21, x21, #16

        .word 0x4f80e110 // sdot v16.4s, v8.16b,  v0.4b[0]
        .word 0x4f80e131 // sdot v17.4s, v9.16b,  v0.4b[0]
        .word 0x4f81e112 // sdot v18.4s, v8.16b,  v1.4b[0]
        .word 0x4f81e133 // sdot v19.4s, v9.16b,  v1.4b[0]
        .word 0x4f82e114 // sdot v20.4s, v8.16b,  v2.4b[0]
        .word 0x4f82e135 // sdot v21.4s, v9.16b,  v2.4b[0]
        .word 0x4f83e116 // sdot v22.4s, v8.16b,  v3.4b[0]
        .word 0x4f83e137 // sdot v23.4s, v9.16b,  v3.4b[0]
        .word 0x4f84e118 // sdot v24.4s, v8.16b,  v4.4b[0]
        .word 0x4f84e139 // sdot v25.4s, v9.16b,  v4.4b[0]
        .word 0x4f85e11a // sdot v26.4s, v8.16b,  v5.4b[0]
        .word 0x4f85e13b // sdot v27.4s, v9.16b,  v5.4b[0]
        .word 0x4f86e11c // sdot v28.4s, v8.16b,  v6.4b[0]
        .word 0x4f86e13d // sdot v29.4s, v9.16b,  v6.4b[0]
        .word 0x4f87e11e // sdot v30.4s, v8.16b,  v7.4b[0]
        .word 0x4f87e13f // sdot v31.4s, v9.16b,  v7.4b[0]

        b LoopCrr16

    LoopCrr16End:

        ldr q12, [x22, #64]
        ldr q13, [x22, #80]
        ldr q14, [x22, #96]
        ldr q15, [x22, #112]
        add x22, x22, #128
        .word 0x4fa0e150 // sdot v16.4s, v10.16b, v0.4b[1]
        .word 0x4fa0e171 // sdot v17.4s, v11.16b, v0.4b[1]
        .word 0x4fa1e152 // sdot v18.4s, v10.16b, v1.4b[1]
        .word 0x4fa1e173 // sdot v19.4s, v11.16b, v1.4b[1]
        .word 0x4fa2e154 // sdot v20.4s, v10.16b, v2.4b[1]
        .word 0x4fa2e175 // sdot v21.4s, v11.16b, v2.4b[1]
        .word 0x4fa3e156 // sdot v22.4s, v10.16b, v3.4b[1]
        .word 0x4fa3e177 // sdot v23.4s, v11.16b, v3.4b[1]
        .word 0x4fa4e158 // sdot v24.4s, v10.16b, v4.4b[1]
        .word 0x4fa4e179 // sdot v25.4s, v11.16b, v4.4b[1]
        .word 0x4fa5e15a // sdot v26.4s, v10.16b, v5.4b[1]
        .word 0x4fa5e17b // sdot v27.4s, v11.16b, v5.4b[1]
        .word 0x4fa6e15c // sdot v28.4s, v10.16b, v6.4b[1]
        .word 0x4fa6e17d // sdot v29.4s, v11.16b, v6.4b[1]
        .word 0x4fa7e15e // sdot v30.4s, v10.16b, v7.4b[1]
        .word 0x4fa7e17f // sdot v31.4s, v11.16b, v7.4b[1]
        .word 0x4f80e990 // sdot v16.4s, v12.16b, v0.4b[2]
        .word 0x4f80e9b1 // sdot v17.4s, v13.16b, v0.4b[2]
        .word 0x4f81e992 // sdot v18.4s, v12.16b, v1.4b[2]
        .word 0x4f81e9b3 // sdot v19.4s, v13.16b, v1.4b[2]
        .word 0x4f82e994 // sdot v20.4s, v12.16b, v2.4b[2]
        .word 0x4f82e9b5 // sdot v21.4s, v13.16b, v2.4b[2]
        .word 0x4f83e996 // sdot v22.4s, v12.16b, v3.4b[2]
        .word 0x4f83e9b7 // sdot v23.4s, v13.16b, v3.4b[2]
        .word 0x4f84e998 // sdot v24.4s, v12.16b, v4.4b[2]
        .word 0x4f84e9b9 // sdot v25.4s, v13.16b, v4.4b[2]
        .word 0x4f85e99a // sdot v26.4s, v12.16b, v5.4b[2]
        .word 0x4f85e9bb // sdot v27.4s, v13.16b, v5.4b[2]
        .word 0x4f86e99c // sdot v28.4s, v12.16b, v6.4b[2]
        .word 0x4f86e9bd // sdot v29.4s, v13.16b, v6.4b[2]
        .word 0x4f87e99e // sdot v30.4s, v12.16b, v7.4b[2]
        .word 0x4f87e9bf // sdot v31.4s, v13.16b, v7.4b[2]
        .word 0x4fa0e9d0 // sdot v16.4s, v14.16b, v0.4b[3]
        .word 0x4fa0e9f1 // sdot v17.4s, v15.16b, v0.4b[3]
        .word 0x4fa1e9d2 // sdot v18.4s, v14.16b, v1.4b[3]
        .word 0x4fa1e9f3 // sdot v19.4s, v15.16b, v1.4b[3]
        .word 0x4fa2e9d4 // sdot v20.4s, v14.16b, v2.4b[3]
        .word 0x4fa2e9f5 // sdot v21.4s, v15.16b, v2.4b[3]
        .word 0x4fa3e9d6 // sdot v22.4s, v14.16b, v3.4b[3]
        .word 0x4fa3e9f7 // sdot v23.4s, v15.16b, v3.4b[3]
        .word 0x4fa4e9d8 // sdot v24.4s, v14.16b, v4.4b[3]
        .word 0x4fa4e9f9 // sdot v25.4s, v15.16b, v4.4b[3]
        .word 0x4fa5e9da // sdot v26.4s, v14.16b, v5.4b[3]
        .word 0x4fa5e9fb // sdot v27.4s, v15.16b, v5.4b[3]
        .word 0x4fa6e9dc // sdot v28.4s, v14.16b, v6.4b[3]
        .word 0x4fa6e9fd // sdot v29.4s, v15.16b, v6.4b[3]
        .word 0x4fa7e9de // sdot v30.4s, v14.16b, v7.4b[3]
        .word 0x4fa7e9ff // sdot v31.4s, v15.16b, v7.4b[3]

    LoopCrr8:
        cmp x21, #7
        ble LoopCrr4

        ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x22], #64

        ld1 {v0.8b}, [x10], #8
        ld1 {v1.8b}, [x11], #8
        ld1 {v2.8b}, [x12], #8
        ld1 {v3.8b}, [x13], #8
        ld1 {v4.8b}, [x14], #8
        ld1 {v5.8b}, [x15], #8
        ld1 {v6.8b}, [x19], #8
        ld1 {v7.8b}, [x20], #8

        .word 0x4f80e110 // sdot v16.4s, v8.16b,  v0.4b[0]
        .word 0x4f80e131 // sdot v17.4s, v9.16b,  v0.4b[0]
        .word 0x4f81e112 // sdot v18.4s, v8.16b,  v1.4b[0]
        .word 0x4f81e133 // sdot v19.4s, v9.16b,  v1.4b[0]
        .word 0x4f82e114 // sdot v20.4s, v8.16b,  v2.4b[0]
        .word 0x4f82e135 // sdot v21.4s, v9.16b,  v2.4b[0]
        .word 0x4f83e116 // sdot v22.4s, v8.16b,  v3.4b[0]
        .word 0x4f83e137 // sdot v23.4s, v9.16b,  v3.4b[0]
        .word 0x4f84e118 // sdot v24.4s, v8.16b,  v4.4b[0]
        .word 0x4f84e139 // sdot v25.4s, v9.16b,  v4.4b[0]
        .word 0x4f85e11a // sdot v26.4s, v8.16b,  v5.4b[0]
        .word 0x4f85e13b // sdot v27.4s, v9.16b,  v5.4b[0]
        .word 0x4f86e11c // sdot v28.4s, v8.16b,  v6.4b[0]
        .word 0x4f86e13d // sdot v29.4s, v9.16b,  v6.4b[0]
        .word 0x4f87e11e // sdot v30.4s, v8.16b,  v7.4b[0]
        .word 0x4f87e13f // sdot v31.4s, v9.16b,  v7.4b[0]

        sub x21, x21, #8

        .word 0x4fa0e150 // sdot v16.4s, v10.16b, v0.4b[1]
        .word 0x4fa0e171 // sdot v17.4s, v11.16b, v0.4b[1]
        .word 0x4fa1e152 // sdot v18.4s, v10.16b, v1.4b[1]
        .word 0x4fa1e173 // sdot v19.4s, v11.16b, v1.4b[1]
        .word 0x4fa2e154 // sdot v20.4s, v10.16b, v2.4b[1]
        .word 0x4fa2e175 // sdot v21.4s, v11.16b, v2.4b[1]
        .word 0x4fa3e156 // sdot v22.4s, v10.16b, v3.4b[1]
        .word 0x4fa3e177 // sdot v23.4s, v11.16b, v3.4b[1]
        .word 0x4fa4e158 // sdot v24.4s, v10.16b, v4.4b[1]
        .word 0x4fa4e179 // sdot v25.4s, v11.16b, v4.4b[1]
        .word 0x4fa5e15a // sdot v26.4s, v10.16b, v5.4b[1]
        .word 0x4fa5e17b // sdot v27.4s, v11.16b, v5.4b[1]
        .word 0x4fa6e15c // sdot v28.4s, v10.16b, v6.4b[1]
        .word 0x4fa6e17d // sdot v29.4s, v11.16b, v6.4b[1]
        .word 0x4fa7e15e // sdot v30.4s, v10.16b, v7.4b[1]
        .word 0x4fa7e17f // sdot v31.4s, v11.16b, v7.4b[1]

        b LoopCrr8
    
    LoopCrr4:
        cmp x21, #3
        ble LoopEnd

        ld1 {v8.16b, v9.16b}, [x22], #32
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
        .word 0x4f80e131 // sdot v17.4s, v9.16b,  v0.4b[0]
        .word 0x4f81e112 // sdot v18.4s, v8.16b,  v1.4b[0]
        .word 0x4f81e133 // sdot v19.4s, v9.16b,  v1.4b[0]
        .word 0x4f82e114 // sdot v20.4s, v8.16b,  v2.4b[0]
        .word 0x4f82e135 // sdot v21.4s, v9.16b,  v2.4b[0]
        .word 0x4f83e116 // sdot v22.4s, v8.16b,  v3.4b[0]
        .word 0x4f83e137 // sdot v23.4s, v9.16b,  v3.4b[0]
        .word 0x4f84e118 // sdot v24.4s, v8.16b,  v4.4b[0]
        .word 0x4f84e139 // sdot v25.4s, v9.16b,  v4.4b[0]
        .word 0x4f85e11a // sdot v26.4s, v8.16b,  v5.4b[0]
        .word 0x4f85e13b // sdot v27.4s, v9.16b,  v5.4b[0]
        .word 0x4f86e11c // sdot v28.4s, v8.16b,  v6.4b[0]
        .word 0x4f86e13d // sdot v29.4s, v9.16b,  v6.4b[0]
        .word 0x4f87e11e // sdot v30.4s, v8.16b,  v7.4b[0]
        .word 0x4f87e13f // sdot v31.4s, v9.16b,  v7.4b[0]

        b LoopCrr4

LoopEnd:
    // hw counter -= 8
    sub x24, x24, #8
    // src_ptr += 8 * src_depth
    add x23, x23, x3, lsl#3

    ldr x10, [sp, #0]   // relu

    // scale oc0 ~ oc7
    ldr q1, [x7]
    ldr q2, [x7, #16]

ConvReluAdd:
    cmp x10, #-1  // if relu == -1, Conv-Relu-Add
    bne MulScale

    eor v0.16b, v0.16b, v0.16b
    smax v16.4s, v16.4s, v0.4s
    smax v17.4s, v17.4s, v0.4s
    smax v18.4s, v18.4s, v0.4s
    smax v19.4s, v19.4s, v0.4s
    smax v20.4s, v20.4s, v0.4s
    smax v21.4s, v21.4s, v0.4s
    smax v22.4s, v22.4s, v0.4s
    smax v23.4s, v23.4s, v0.4s
    smax v24.4s, v24.4s, v0.4s
    smax v25.4s, v25.4s, v0.4s
    smax v26.4s, v26.4s, v0.4s
    smax v27.4s, v27.4s, v0.4s
    smax v28.4s, v28.4s, v0.4s
    smax v29.4s, v29.4s, v0.4s
    smax v30.4s, v30.4s, v0.4s
    smax v31.4s, v31.4s, v0.4s
MulScale:
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    scvtf v24.4s, v24.4s
    scvtf v25.4s, v25.4s
    scvtf v26.4s, v26.4s
    scvtf v27.4s, v27.4s
    scvtf v28.4s, v28.4s
    scvtf v29.4s, v29.4s
    scvtf v30.4s, v30.4s
    scvtf v31.4s, v31.4s

    fmul v16.4s, v16.4s, v1.4s
    fmul v17.4s, v17.4s, v2.4s
    fmul v18.4s, v18.4s, v1.4s
    fmul v19.4s, v19.4s, v2.4s
    fmul v20.4s, v20.4s, v1.4s
    fmul v21.4s, v21.4s, v2.4s
    fmul v22.4s, v22.4s, v1.4s
    fmul v23.4s, v23.4s, v2.4s
    fmul v24.4s, v24.4s, v1.4s
    fmul v25.4s, v25.4s, v2.4s
    fmul v26.4s, v26.4s, v1.4s
    fmul v27.4s, v27.4s, v2.4s
    fmul v28.4s, v28.4s, v1.4s
    fmul v29.4s, v29.4s, v2.4s
    fmul v30.4s, v30.4s, v1.4s
    fmul v31.4s, v31.4s, v2.4s

    cbz x26, ConvAddPost  // if add_input == 0, skip

AddInputScale:

    // add_input_ptr 0 ~ 7
    add x11, x28, x4
    add x12, x28, x4, lsl#1
    add x14, x28, x4, lsl#2
    add x13, x11, x4, lsl#1
    add x15, x11, x4, lsl#2
    add x19, x12, x4, lsl#2
    add x20, x13, x4, lsl#2
    ldr d0, [x28]
    ldr d2, [x11]
    ldr d4, [x12]
    ldr d6, [x13]
    ldr d8, [x14]
    ldr d9, [x15]
    ldr d10, [x19]
    ldr d11, [x20]
    // add_scale
    ldr q12, [x27]
    ldr q13, [x27, #16]

    // convert add_input int8 to fp32
    sxtl v0.8h, v0.8b
    sxtl v2.8h, v2.8b
    sxtl v4.8h, v4.8b
    sxtl v6.8h, v6.8b
    sxtl2 v1.4s, v0.8h
    sxtl2 v3.4s, v2.8h
    sxtl2 v5.4s, v4.8h
    sxtl2 v7.4s, v6.8h
    sxtl  v0.4s, v0.4h
    sxtl  v2.4s, v2.4h
    sxtl  v4.4s, v4.4h
    sxtl  v6.4s, v6.4h
    scvtf v0.4s, v0.4s
    scvtf v1.4s, v1.4s
    scvtf v2.4s, v2.4s
    scvtf v3.4s, v3.4s
    scvtf v4.4s, v4.4s
    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    scvtf v7.4s, v7.4s

    fmla v16.4s, v0.4s, v12.4s
    sxtl v0.8h, v8.8b
    fmla v17.4s, v1.4s, v13.4s
    fmla v18.4s, v2.4s, v12.4s
    sxtl v2.8h, v9.8b
    fmla v19.4s, v3.4s, v13.4s
    fmla v20.4s, v4.4s, v12.4s
    sxtl v4.8h, v10.8b
    fmla v21.4s, v5.4s, v13.4s
    fmla v22.4s, v6.4s, v12.4s
    sxtl v6.8h, v11.8b
    fmla v23.4s, v7.4s, v13.4s

    sxtl2 v1.4s, v0.8h
    sxtl2 v3.4s, v2.8h
    sxtl2 v5.4s, v4.8h
    sxtl2 v7.4s, v6.8h
    sxtl  v0.4s, v0.4h
    sxtl  v2.4s, v2.4h
    sxtl  v4.4s, v4.4h
    sxtl  v6.4s, v6.4h
    scvtf v0.4s, v0.4s
    scvtf v1.4s, v1.4s
    scvtf v2.4s, v2.4s
    scvtf v3.4s, v3.4s
    scvtf v4.4s, v4.4s
    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    scvtf v7.4s, v7.4s

    fmla v24.4s, v0.4s, v12.4s
    fmla v25.4s, v1.4s, v13.4s
    fmla v26.4s, v2.4s, v12.4s
    fmla v27.4s, v3.4s, v13.4s
    fmla v28.4s, v4.4s, v12.4s
    fmla v29.4s, v5.4s, v13.4s
    fmla v30.4s, v6.4s, v12.4s
    fmla v31.4s, v7.4s, v13.4s

    // add_input_ptr += 8 * dst_depth
    add x28, x28, x4, lsl#3

ConvAddPost:
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s
    fcvtas v22.4s, v22.4s
    fcvtas v23.4s, v23.4s
    fcvtas v24.4s, v24.4s
    fcvtas v25.4s, v25.4s
    fcvtas v26.4s, v26.4s
    fcvtas v27.4s, v27.4s
    fcvtas v28.4s, v28.4s
    fcvtas v29.4s, v29.4s
    fcvtas v30.4s, v30.4s
    fcvtas v31.4s, v31.4s

    sqxtn  v16.4h, v16.4s
    sqxtn  v18.4h, v18.4s
    sqxtn  v20.4h, v20.4s
    sqxtn  v22.4h, v22.4s
    sqxtn  v24.4h, v24.4s
    sqxtn  v26.4h, v26.4s
    sqxtn  v28.4h, v28.4s
    sqxtn  v30.4h, v30.4s
    sqxtn2 v16.8h, v17.4s
    sqxtn2 v18.8h, v19.4s
    sqxtn2 v20.8h, v21.4s
    sqxtn2 v22.8h, v23.4s
    sqxtn2 v24.8h, v25.4s
    sqxtn2 v26.8h, v27.4s
    sqxtn2 v28.8h, v29.4s
    sqxtn2 v30.8h, v31.4s

    sqxtn v16.8b, v16.8h
    sqxtn v18.8b, v18.8h
    sqxtn v20.8b, v20.8h
    sqxtn v22.8b, v22.8h
    sqxtn v24.8b, v24.8h
    sqxtn v26.8b, v26.8h
    sqxtn v28.8b, v28.8h
    sqxtn v30.8b, v30.8h

    cmp x10, #1  // if relu != 1 or 2, Conv-Add-Relu or Relu6, skip
    blt ConvAddPostEnd
    eor v0.16b, v0.16b, v0.16b
    smax v16.8b, v16.8b, v0.8b
    smax v18.8b, v18.8b, v0.8b
    smax v20.8b, v20.8b, v0.8b
    smax v22.8b, v22.8b, v0.8b
    smax v24.8b, v24.8b, v0.8b
    smax v26.8b, v26.8b, v0.8b
    smax v28.8b, v28.8b, v0.8b
    smax v30.8b, v30.8b, v0.8b

    cmp x10, #2   // relu6
    bne ConvAddPostEnd
    ldr x13, [sp, #24]  // relu6_max
    ld1 {v0.8b}, [x13]
    smin v16.8b, v16.8b, v0.8b
    smin v18.8b, v18.8b, v0.8b
    smin v20.8b, v20.8b, v0.8b
    smin v22.8b, v22.8b, v0.8b
    smin v24.8b, v24.8b, v0.8b
    smin v26.8b, v26.8b, v0.8b
    smin v28.8b, v28.8b, v0.8b
    smin v30.8b, v30.8b, v0.8b

ConvAddPostEnd:

    // store to dst_ptr 0 ~ 7
    mov x10, x25
    add x11, x25, x4
    add x12, x25, x4, lsl#1
    add x14, x25, x4, lsl#2
    add x13, x11, x4, lsl#1
    add x15, x11, x4, lsl#2
    add x19, x12, x4, lsl#2
    add x20, x13, x4, lsl#2

    str d16, [x10]
    str d18, [x11]
    str d20, [x12]
    str d22, [x13]
    str d24, [x14]
    str d26, [x15]
    str d28, [x19]
    str d30, [x20]

    // dst_ptr += 8 * dst_depth
    add x25, x25, x4, lsl#3

    b LoopHW8

LoopHW1:
    // if hw counter <= 0, skip
    cmp x24, #0
    ble LoopHW1End

    // src_ptr_0
    mov x10, x23

    // load bias 32bit, accumulator 2 reg
    ld1 {v16.4s, v17.4s}, [x6]

    // src_depth counter
    mov x21, x3

    // weight_ptr
    mov x22, x2

    HW1LoopCrr16:
        cmp x21, #15
        ble HW1LoopCrr8

        ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x22], #64
        ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x22], #64
        ld1 {v0.16b}, [x10], #16
        sub x21, x21, #16
        .word 0x4f80e110 // sdot v16.4s, v8.16b,  v0.4b[0]
        .word 0x4f80e131 // sdot v17.4s, v9.16b,  v0.4b[0]
        .word 0x4fa0e150 // sdot v16.4s, v10.16b, v0.4b[1]
        .word 0x4fa0e171 // sdot v17.4s, v11.16b, v0.4b[1]
        .word 0x4f80e990 // sdot v16.4s, v12.16b, v0.4b[2]
        .word 0x4f80e9b1 // sdot v17.4s, v13.16b, v0.4b[2]
        .word 0x4fa0e9d0 // sdot v16.4s, v14.16b, v0.4b[3]
        .word 0x4fa0e9f1 // sdot v17.4s, v15.16b, v0.4b[3]
        b HW1LoopCrr16
    
    HW1LoopCrr8:
        cmp x21, #7
        ble HW1LoopCrr4

        ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x22], #64
        ld1 {v0.8b}, [x10], #8
        sub x21, x21, #8
        .word 0x4f80e110 // sdot v16.4s, v8.16b,  v0.4b[0]
        .word 0x4f80e131 // sdot v17.4s, v9.16b,  v0.4b[0]
        .word 0x4fa0e150 // sdot v16.4s, v10.16b, v0.4b[1]
        .word 0x4fa0e171 // sdot v17.4s, v11.16b, v0.4b[1]
        b HW1LoopCrr8

    HW1LoopCrr4:
        cmp x21, #3
        ble HW1LoopEnd

        ld1 {v8.16b, v9.16b}, [x22], #32
        ld1 {v0.s}[0], [x10], #4
        sub x21, x21, #4
        .word 0x4f80e110 // sdot v16.4s, v8.16b,  v0.4b[0]
        .word 0x4f80e131 // sdot v17.4s, v9.16b,  v0.4b[0]
        b HW1LoopCrr4

HW1LoopEnd:
    // hw counter -= 1
    sub x24, x24, #1
    // src_ptr += 1 * src_depth
    add x23, x23, x3

    ldr x10, [sp, #0]   // relu

    // scale oc0 ~ oc7
    ldr q1, [x7]
    ldr q2, [x7, #16]

HW1ConvReluAdd:
    cmp x10, #-1  // if relu == -1, Conv-Relu-Add
    bne HW1MulScale

    eor v0.16b, v0.16b, v0.16b
    smax v16.4s, v16.4s, v0.4s
    smax v17.4s, v17.4s, v0.4s

HW1MulScale:
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s

    fmul v16.4s, v16.4s, v1.4s
    fmul v17.4s, v17.4s, v2.4s

    cbz x26, HW1ConvAddPost  // if add_input == 0, skip

HW1AddInputScale:

    ldr d0, [x28]
    // add_scale
    ldr q12, [x27]
    ldr q13, [x27, #16]

    // convert add_input int8 to fp32
    sxtl v0.8h, v0.8b
    sxtl  v1.4s, v0.4h
    sxtl2 v2.4s, v0.8h
    scvtf v1.4s, v1.4s
    scvtf v2.4s, v2.4s
    fmla v16.4s, v1.4s, v12.4s
    fmla v17.4s, v2.4s, v13.4s

    // add_input_ptr += 1 * dst_depth
    add x28, x28, x4

HW1ConvAddPost:
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    sqxtn  v16.4h, v16.4s
    sqxtn2 v16.8h, v17.4s
    sqxtn v16.8b, v16.8h

    cmp x10, #1  // if relu != 1 or 2, Conv-Add-Relu or Relu6, skip
    blt HW1ConvAddPostEnd
    eor v0.16b, v0.16b, v0.16b
    smax v16.8b, v16.8b, v0.8b

    cmp x10, #2   // relu6
    bne HW1ConvAddPostEnd
    ldr x13, [sp, #24]  // relu6_max
    ld1 {v0.8b}, [x13]
    smin v16.8b, v16.8b, v0.8b

HW1ConvAddPostEnd:
    // store to dst_ptr
    str d16, [x25]
    // dst_ptr += 1 * dst_depth
    add x25, x25, x4

    b LoopHW1

LoopHW1End:

// dst += 8 * sizeof(int8)
add x0, x0, #8
// bias += 8 * sizeof(int32)
add x6, x6, #32
// weight += 8 * src_depth
add x2, x2, x3, lsl#3
// scale += 8 * sizeof(fp32)
add x7, x7, #32
// relu6_max += 8 * sizeof(int32)
ldr x13, [sp, #24]
add x13, x13, #8
str x13, [sp, #24]
// if add_input != 0
// add_input += 8 * sizeof(int8)
// add_scale ++ 8 * sizeof(fp32)
cbz x26, UpdateAddInputEnd
add x26, x26, #8
add x27, x27, #32
UpdateAddInputEnd:

subs x9, x9, #8
bge LoopDz8

sub sp, sp, #208
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ldp x19, x20, [sp], #16
ldp x21, x22, [sp], #16
ldp x23, x24, [sp], #16
ldp x25, x26, [sp], #16
ldp x27, x28, [sp], #16

END:
ret

#endif
#endif
