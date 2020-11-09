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

#ifdef __aarch64__

#include "tnn/device/arm/acc/compute/asm_func_name.S"

.text
.align 5
asm_function GemmInt8Unit8x8
//void GemmInt8Unit8x8(long mr, long nr, long k, const int8_t* a, long a_stride, const void* w, int8_t* c, long c_stride,
//                     const float* scales, long relu)
//x0(mr),
//x1(nr),
//x2(k),
//x3(src),
//x4(a_stride),
//x5(weight),
//x6(dst),
//x7(c_stride)
//from stack(scale)

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

ldr x8, [sp, #0]

// load bias 32bit, accumulator 16 reg
ld1 {v8.4s, v9.4s}, [x5], #32
mov v10.16b, v8.16b
mov v11.16b, v9.16b
mov v12.16b, v8.16b
mov v13.16b, v9.16b
mov v14.16b, v8.16b
mov v15.16b, v9.16b
mov v16.16b, v8.16b
mov v17.16b, v9.16b
mov v18.16b, v8.16b
mov v19.16b, v9.16b
mov v20.16b, v8.16b
mov v21.16b, v9.16b
mov v22.16b, v8.16b
mov v23.16b, v9.16b

# a1
cmp x0, #2
add x9, x3, x4
csel x9, x3, x9, lo
# a2
add x10, x9, x4
csel x10, x9, x10, ls
# a3
cmp x0, #4
add x11, x10, x4
csel x11, x10, x11, lo
# a4
add x12, x11, x4
csel x12, x11, x12, ls
# a5
cmp x0, #6
add x13, x12, x4
csel x13, x12, x13, lo
# a6
add x14, x13, x4
csel x14, x13, x14, ls
# a7
cmp x0, #8
add x15, x14, x4
csel x15, x14, x15, ne 

subs x2, x2, #8
blo 1f

0:
    ld1 {v27.8b}, [x5], #8
    sxtl v27.8h, v27.8b
    ld1 {v0.8b}, [x3], #8
    sxtl v0.8h, v0.8b
    ld1 {v1.8b}, [x9], #8
    sxtl v1.8h, v1.8b
    ld1 {v2.8b}, [x10], #8
    sxtl v2.8h, v2.8b
    ld1 {v3.8b}, [x11], #8
    sxtl v3.8h, v3.8b
    ld1 {v4.8b}, [x12], #8
    sxtl v4.8h, v4.8b
    ld1 {v5.8b}, [x13], #8
    sxtl v5.8h, v5.8b
    ld1 {v6.8b}, [x14], #8
    sxtl v6.8h, v6.8b
    ld1 {v7.8b}, [x15], #8
    sxtl v7.8h, v7.8b

    // c0
    ld1 {v28.8b}, [x5], #8
    smlal v8.4s, v27.4h, v0.h[0]
    smlal2 v9.4s, v27.8h, v0.h[0]
    smlal v10.4s, v27.4h, v1.h[0]
    smlal2 v11.4s, v27.8h, v1.h[0]
    smlal v12.4s, v27.4h, v2.h[0]
    smlal2 v13.4s, v27.8h, v2.h[0]
    smlal v14.4s, v27.4h, v3.h[0]
    smlal2 v15.4s, v27.8h, v3.h[0]
    sxtl v28.8h, v28.8b
    smlal v16.4s, v27.4h, v4.h[0]
    smlal2 v17.4s, v27.8h, v4.h[0]
    smlal v18.4s, v27.4h, v5.h[0]
    smlal2 v19.4s, v27.8h, v5.h[0]
    smlal v20.4s, v27.4h, v6.h[0]
    smlal2 v21.4s, v27.8h, v6.h[0]
    smlal v22.4s, v27.4h, v7.h[0]
    smlal2 v23.4s, v27.8h, v7.h[0]

    // c1            
    ld1 {v27.8b}, [x5], #8
    smlal v8.4s, v28.4h, v0.h[1]
    smlal2 v9.4s, v28.8h, v0.h[1]
    smlal v10.4s, v28.4h, v1.h[1]
    smlal2 v11.4s, v28.8h, v1.h[1]
    smlal v12.4s, v28.4h, v2.h[1]
    smlal2 v13.4s, v28.8h, v2.h[1]
    smlal v14.4s, v28.4h, v3.h[1]
    smlal2 v15.4s, v28.8h, v3.h[1]
    sxtl v27.8h, v27.8b
    smlal v16.4s, v28.4h, v4.h[1]
    smlal2 v17.4s, v28.8h, v4.h[1]
    smlal v18.4s, v28.4h, v5.h[1]
    smlal2 v19.4s, v28.8h, v5.h[1]
    smlal v20.4s, v28.4h, v6.h[1]
    smlal2 v21.4s, v28.8h, v6.h[1]
    smlal v22.4s, v28.4h, v7.h[1]
    smlal2 v23.4s, v28.8h, v7.h[1]

    // c2            
    ld1 {v28.8b}, [x5], #8
    smlal v8.4s, v27.4h, v0.h[2]
    smlal2 v9.4s, v27.8h, v0.h[2]
    smlal v10.4s, v27.4h, v1.h[2]
    smlal2 v11.4s, v27.8h, v1.h[2]
    smlal v12.4s, v27.4h, v2.h[2]
    smlal2 v13.4s, v27.8h, v2.h[2]
    smlal v14.4s, v27.4h, v3.h[2]
    smlal2 v15.4s, v27.8h, v3.h[2]
    sxtl v28.8h, v28.8b
    smlal v16.4s, v27.4h, v4.h[2]
    smlal2 v17.4s, v27.8h, v4.h[2]
    smlal v18.4s, v27.4h, v5.h[2]
    smlal2 v19.4s, v27.8h, v5.h[2]
    smlal v20.4s, v27.4h, v6.h[2]
    smlal2 v21.4s, v27.8h, v6.h[2]
    smlal v22.4s, v27.4h, v7.h[2]
    smlal2 v23.4s, v27.8h, v7.h[2]

    // c3            
    ld1 {v27.8b}, [x5], #8
    smlal v8.4s, v28.4h, v0.h[3]
    smlal2 v9.4s, v28.8h, v0.h[3]
    smlal v10.4s, v28.4h, v1.h[3]
    smlal2 v11.4s, v28.8h, v1.h[3]
    smlal v12.4s, v28.4h, v2.h[3]
    smlal2 v13.4s, v28.8h, v2.h[3]
    smlal v14.4s, v28.4h, v3.h[3]
    smlal2 v15.4s, v28.8h, v3.h[3]
    sxtl v27.8h, v27.8b
    smlal v16.4s, v28.4h, v4.h[3]
    smlal2 v17.4s, v28.8h, v4.h[3]
    smlal v18.4s, v28.4h, v5.h[3]
    smlal2 v19.4s, v28.8h, v5.h[3]
    smlal v20.4s, v28.4h, v6.h[3]
    smlal2 v21.4s, v28.8h, v6.h[3]
    smlal v22.4s, v28.4h, v7.h[3]
    smlal2 v23.4s, v28.8h, v7.h[3]

    // c4
    ld1 {v28.8b}, [x5], #8
    smlal v8.4s, v27.4h, v0.h[4]
    smlal2 v9.4s, v27.8h, v0.h[4]
    smlal v10.4s, v27.4h, v1.h[4]
    smlal2 v11.4s, v27.8h, v1.h[4]
    smlal v12.4s, v27.4h, v2.h[4]
    smlal2 v13.4s, v27.8h, v2.h[4]
    smlal v14.4s, v27.4h, v3.h[4]
    smlal2 v15.4s, v27.8h, v3.h[4]
    sxtl v28.8h, v28.8b
    smlal v16.4s, v27.4h, v4.h[4]
    smlal2 v17.4s, v27.8h, v4.h[4]
    smlal v18.4s, v27.4h, v5.h[4]
    smlal2 v19.4s, v27.8h, v5.h[4]
    smlal v20.4s, v27.4h, v6.h[4]
    smlal2 v21.4s, v27.8h, v6.h[4]
    smlal v22.4s, v27.4h, v7.h[4]
    smlal2 v23.4s, v27.8h, v7.h[4]

    // c5            
    ld1 {v27.8b}, [x5], #8
    smlal v8.4s, v28.4h, v0.h[5]
    smlal2 v9.4s, v28.8h, v0.h[5]
    smlal v10.4s, v28.4h, v1.h[5]
    smlal2 v11.4s, v28.8h, v1.h[5]
    smlal v12.4s, v28.4h, v2.h[5]
    smlal2 v13.4s, v28.8h, v2.h[5]
    smlal v14.4s, v28.4h, v3.h[5]
    smlal2 v15.4s, v28.8h, v3.h[5]
    sxtl v27.8h, v27.8b
    smlal v16.4s, v28.4h, v4.h[5]
    smlal2 v17.4s, v28.8h, v4.h[5]
    smlal v18.4s, v28.4h, v5.h[5]
    smlal2 v19.4s, v28.8h, v5.h[5]
    smlal v20.4s, v28.4h, v6.h[5]
    smlal2 v21.4s, v28.8h, v6.h[5]
    smlal v22.4s, v28.4h, v7.h[5]
    smlal2 v23.4s, v28.8h, v7.h[5]

    // c6            
    ld1 {v28.8b}, [x5], #8
    smlal v8.4s, v27.4h, v0.h[6]
    smlal2 v9.4s, v27.8h, v0.h[6]
    smlal v10.4s, v27.4h, v1.h[6]
    smlal2 v11.4s, v27.8h, v1.h[6]
    smlal v12.4s, v27.4h, v2.h[6]
    smlal2 v13.4s, v27.8h, v2.h[6]
    smlal v14.4s, v27.4h, v3.h[6]
    smlal2 v15.4s, v27.8h, v3.h[6]
    sxtl v28.8h, v28.8b
    smlal v16.4s, v27.4h, v4.h[6]
    smlal2 v17.4s, v27.8h, v4.h[6]
    smlal v18.4s, v27.4h, v5.h[6]
    smlal2 v19.4s, v27.8h, v5.h[6]
    smlal v20.4s, v27.4h, v6.h[6]
    smlal2 v21.4s, v27.8h, v6.h[6]
    smlal v22.4s, v27.4h, v7.h[6]
    smlal2 v23.4s, v27.8h, v7.h[6]

    // c7            
    subs x2, x2, #8

    smlal v8.4s, v28.4h, v0.h[7]
    smlal2 v9.4s, v28.8h, v0.h[7]
    smlal v10.4s, v28.4h, v1.h[7]
    smlal2 v11.4s, v28.8h, v1.h[7]
    smlal v12.4s, v28.4h, v2.h[7]
    smlal2 v13.4s, v28.8h, v2.h[7]
    smlal v14.4s, v28.4h, v3.h[7]
    smlal2 v15.4s, v28.8h, v3.h[7]
    smlal v16.4s, v28.4h, v4.h[7]
    smlal2 v17.4s, v28.8h, v4.h[7]
    smlal v18.4s, v28.4h, v5.h[7]
    smlal2 v19.4s, v28.8h, v5.h[7]
    smlal v20.4s, v28.4h, v6.h[7]
    smlal2 v21.4s, v28.8h, v6.h[7]
    smlal v22.4s, v28.4h, v7.h[7]
    smlal2 v23.4s, v28.8h, v7.h[7]

    bhs 0b

1:
    cmp x2, #-8
    beq 2f

    // Adjust a0-a7
    add x3, x3, x2
    add x9, x9, x2
    add x10, x10, x2
    add x11, x11, x2
    add x12, x12, x2
    add x13, x13, x2
    add x14, x14, x2
    add x15, x15, x2

    lsl x2, x2, #3
    fmov d29, x2

    // Load x0-a7
    ld1 {v0.8b}, [x3], #8
    sshl d0, d0, d29
    sxtl v0.8h, v0.8b

    ld1 {v1.8b}, [x9], #8
    sshl d1, d1, d29
    sxtl v1.8h, v1.8b

    ld1 {v2.8b}, [x10], #8
    sshl d2, d2, d29
    sxtl v2.8h, v2.8b

    ld1 {v3.8b}, [x11], #8
    sshl d3, d3, d29
    sxtl v3.8h, v3.8b

    ld1 {v4.8b}, [x12], #8
    sshl d4, d4, d29
    sxtl v4.8h, v4.8b

    ld1 {v5.8b}, [x13], #8
    sshl d5, d5, d29
    sxtl v5.8h, v5.8b

    ld1 {v6.8b}, [x14], #8
    sshl d6, d6, d29
    sxtl v6.8h, v6.8b

    ld1 {v7.8b}, [x15], #8
    sshl d7, d7, d29
    sxtl v7.8h, v7.8b

    // c0
    ld1 {v27.8b}, [x5], #8
    sxtl v27.8h, v27.8b

    smlal v8.4s, v27.4h, v0.h[0]
    smlal2 v9.4s, v27.8h, v0.h[0]
    smlal v10.4s, v27.4h, v1.h[0]
    smlal2 v11.4s, v27.8h, v1.h[0]
    smlal v12.4s, v27.4h, v2.h[0]
    smlal2 v13.4s, v27.8h, v2.h[0]
    smlal v14.4s, v27.4h, v3.h[0]
    smlal2 v15.4s, v27.8h, v3.h[0]
    smlal v16.4s, v27.4h, v4.h[0]
    smlal2 v17.4s, v27.8h, v4.h[0]
    smlal v18.4s, v27.4h, v5.h[0]
    smlal2 v19.4s, v27.8h, v5.h[0]
    smlal v20.4s, v27.4h, v6.h[0]
    smlal2 v21.4s, v27.8h, v6.h[0]
    smlal v22.4s, v27.4h, v7.h[0]
    smlal2 v23.4s, v27.8h, v7.h[0]

    cmp x2, #-48
    blo 2f

    // c1            
    ld1 {v28.8b}, [x5], #8
    sxtl v28.8h, v28.8b

    smlal v8.4s, v28.4h, v0.h[1]
    smlal2 v9.4s, v28.8h, v0.h[1]
    smlal v10.4s, v28.4h, v1.h[1]
    smlal2 v11.4s, v28.8h, v1.h[1]
    smlal v12.4s, v28.4h, v2.h[1]
    smlal2 v13.4s, v28.8h, v2.h[1]
    smlal v14.4s, v28.4h, v3.h[1]
    smlal2 v15.4s, v28.8h, v3.h[1]
    smlal v16.4s, v28.4h, v4.h[1]
    smlal2 v17.4s, v28.8h, v4.h[1]
    smlal v18.4s, v28.4h, v5.h[1]
    smlal2 v19.4s, v28.8h, v5.h[1]
    smlal v20.4s, v28.4h, v6.h[1]
    smlal2 v21.4s, v28.8h, v6.h[1]
    smlal v22.4s, v28.4h, v7.h[1]
    smlal2 v23.4s, v28.8h, v7.h[1]

    bls 2f

    // c2            
    ld1 {v27.8b}, [x5], #8
    sxtl v27.8h, v27.8b

    smlal v8.4s, v27.4h, v0.h[2]
    smlal2 v9.4s, v27.8h, v0.h[2]
    smlal v10.4s, v27.4h, v1.h[2]
    smlal2 v11.4s, v27.8h, v1.h[2]
    smlal v12.4s, v27.4h, v2.h[2]
    smlal2 v13.4s, v27.8h, v2.h[2]
    smlal v14.4s, v27.4h, v3.h[2]
    smlal2 v15.4s, v27.8h, v3.h[2]
    smlal v16.4s, v27.4h, v4.h[2]
    smlal2 v17.4s, v27.8h, v4.h[2]
    smlal v18.4s, v27.4h, v5.h[2]
    smlal2 v19.4s, v27.8h, v5.h[2]
    smlal v20.4s, v27.4h, v6.h[2]
    smlal2 v21.4s, v27.8h, v6.h[2]
    smlal v22.4s, v27.4h, v7.h[2]
    smlal2 v23.4s, v27.8h, v7.h[2]

    cmp x2, #-32
    blo 2f

    // c3            
    ld1 {v28.8b}, [x5], #8
    sxtl v28.8h, v28.8b

    smlal v8.4s, v28.4h, v0.h[3]
    smlal2 v9.4s, v28.8h, v0.h[3]
    smlal v10.4s, v28.4h, v1.h[3]
    smlal2 v11.4s, v28.8h, v1.h[3]
    smlal v12.4s, v28.4h, v2.h[3]
    smlal2 v13.4s, v28.8h, v2.h[3]
    smlal v14.4s, v28.4h, v3.h[3]
    smlal2 v15.4s, v28.8h, v3.h[3]
    smlal v16.4s, v28.4h, v4.h[3]
    smlal2 v17.4s, v28.8h, v4.h[3]
    smlal v18.4s, v28.4h, v5.h[3]
    smlal2 v19.4s, v28.8h, v5.h[3]
    smlal v20.4s, v28.4h, v6.h[3]
    smlal2 v21.4s, v28.8h, v6.h[3]
    smlal v22.4s, v28.4h, v7.h[3]
    smlal2 v23.4s, v28.8h, v7.h[3]

    bls 2f

    // c4
    ld1 {v27.8b}, [x5], #8
    sxtl v27.8h, v27.8b

    smlal v8.4s, v27.4h, v0.h[4]
    smlal2 v9.4s, v27.8h, v0.h[4]
    smlal v10.4s, v27.4h, v1.h[4]
    smlal2 v11.4s, v27.8h, v1.h[4]
    smlal v12.4s, v27.4h, v2.h[4]
    smlal2 v13.4s, v27.8h, v2.h[4]
    smlal v14.4s, v27.4h, v3.h[4]
    smlal2 v15.4s, v27.8h, v3.h[4]
    smlal v16.4s, v27.4h, v4.h[4]
    smlal2 v17.4s, v27.8h, v4.h[4]
    smlal v18.4s, v27.4h, v5.h[4]
    smlal2 v19.4s, v27.8h, v5.h[4]
    smlal v20.4s, v27.4h, v6.h[4]
    smlal2 v21.4s, v27.8h, v6.h[4]
    smlal v22.4s, v27.4h, v7.h[4]
    smlal2 v23.4s, v27.8h, v7.h[4]

    cmp x2, #-16
    blo 2f

    // c5            
    ld1 {v28.8b}, [x5], #8
    sxtl v28.8h, v28.8b

    smlal v8.4s, v28.4h, v0.h[5]
    smlal2 v9.4s, v28.8h, v0.h[5]
    smlal v10.4s, v28.4h, v1.h[5]
    smlal2 v11.4s, v28.8h, v1.h[5]
    smlal v12.4s, v28.4h, v2.h[5]
    smlal2 v13.4s, v28.8h, v2.h[5]
    smlal v14.4s, v28.4h, v3.h[5]
    smlal2 v15.4s, v28.8h, v3.h[5]
    smlal v16.4s, v28.4h, v4.h[5]
    smlal2 v17.4s, v28.8h, v4.h[5]
    smlal v18.4s, v28.4h, v5.h[5]
    smlal2 v19.4s, v28.8h, v5.h[5]
    smlal v20.4s, v28.4h, v6.h[5]
    smlal2 v21.4s, v28.8h, v6.h[5]
    smlal v22.4s, v28.4h, v7.h[5]
    smlal2 v23.4s, v28.8h, v7.h[5]

    bls 2f

    // c6            
    ld1 {v27.8b}, [x5], #8
    sxtl v27.8h, v27.8b

    smlal v8.4s, v27.4h, v0.h[6]
    smlal2 v9.4s, v27.8h, v0.h[6]
    smlal v10.4s, v27.4h, v1.h[6]
    smlal2 v11.4s, v27.8h, v1.h[6]
    smlal v12.4s, v27.4h, v2.h[6]
    smlal2 v13.4s, v27.8h, v2.h[6]
    smlal v14.4s, v27.4h, v3.h[6]
    smlal2 v15.4s, v27.8h, v3.h[6]
    smlal v16.4s, v27.4h, v4.h[6]
    smlal2 v17.4s, v27.8h, v4.h[6]
    smlal v18.4s, v27.4h, v5.h[6]
    smlal2 v19.4s, v27.8h, v5.h[6]
    smlal v20.4s, v27.4h, v6.h[6]
    smlal2 v21.4s, v27.8h, v6.h[6]
    smlal v22.4s, v27.4h, v7.h[6]
    smlal2 v23.4s, v27.8h, v7.h[6]

2:
    ld1 {v24.4s}, [x8], #16
    movi v25.4s, #0
    cmp x1, #4
    ble 22f
    ld1 {v25.4s}, [x8]
22:
    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s
    scvtf v10.4s, v10.4s
    scvtf v11.4s, v11.4s
    scvtf v12.4s, v12.4s
    scvtf v13.4s, v13.4s
    scvtf v14.4s, v14.4s
    scvtf v15.4s, v15.4s
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s

    fmul v8.4s, v8.4s, v24.4s
    fmul v10.4s, v10.4s, v24.4s
    fmul v12.4s, v12.4s, v24.4s
    fmul v14.4s, v14.4s, v24.4s
    fmul v16.4s, v16.4s, v24.4s
    fmul v18.4s, v18.4s, v24.4s
    fmul v20.4s, v20.4s, v24.4s
    fmul v22.4s, v22.4s, v24.4s
    fmul v9.4s, v9.4s, v25.4s
    fmul v11.4s, v11.4s, v25.4s
    fmul v13.4s, v13.4s, v25.4s
    fmul v15.4s, v15.4s, v25.4s
    fmul v17.4s, v17.4s, v25.4s
    fmul v19.4s, v19.4s, v25.4s
    fmul v21.4s, v21.4s, v25.4s
    fmul v23.4s, v23.4s, v25.4s

    fcvtas v8.4s, v8.4s
    fcvtas v9.4s, v9.4s
    fcvtas v10.4s, v10.4s
    fcvtas v11.4s, v11.4s
    fcvtas v12.4s, v12.4s
    fcvtas v13.4s, v13.4s
    fcvtas v14.4s, v14.4s
    fcvtas v15.4s, v15.4s
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s
    fcvtas v22.4s, v22.4s
    fcvtas v23.4s, v23.4s

    sqxtn v8.4h, v8.4s
    sqxtn v10.4h, v10.4s
    sqxtn v12.4h, v12.4s
    sqxtn v14.4h, v14.4s
    sqxtn v16.4h, v16.4s
    sqxtn v18.4h, v18.4s
    sqxtn v20.4h, v20.4s
    sqxtn v22.4h, v22.4s
    sqxtn2 v8.8h, v9.4s
    sqxtn2 v10.8h, v11.4s
    sqxtn2 v12.8h, v13.4s
    sqxtn2 v14.8h, v15.4s
    sqxtn2 v16.8h, v17.4s
    sqxtn2 v18.8h, v19.4s
    sqxtn2 v20.8h, v21.4s
    sqxtn2 v22.8h, v23.4s

    sqxtn v8.8b, v8.8h
    sqxtn v10.8b, v10.8h
    sqxtn v12.8b, v12.8h
    sqxtn v14.8b, v14.8h
    sqxtn v16.8b, v16.8h
    sqxtn v18.8b, v18.8h
    sqxtn v20.8b, v20.8h
    sqxtn v22.8b, v22.8h

    ldr x8, [sp, #8]
    cbz x8, 3f
    movi v0.16b, #0
    smax v8.8b, v8.8b, v0.8b
    smax v10.8b, v10.8b, v0.8b
    smax v12.8b, v12.8b, v0.8b
    smax v14.8b, v14.8b, v0.8b
    smax v16.8b, v16.8b, v0.8b
    smax v18.8b, v18.8b, v0.8b
    smax v20.8b, v20.8b, v0.8b
    smax v22.8b, v22.8b, v0.8b

3:
    add  x9, x6,  x7
    cmp x0, #2
    csel x9, x6, x9, lo

    add x10, x9,  x7
    csel x10, x9, x10, ls

    add x11, x10, x7
    cmp x0, #4
    csel x11, x10, x11, lo

    add x12, x11, x7
    csel x12, x11, x12, ls

    add x13, x12, x7
    cmp x0, #6
    csel x13, x12, x13, lo

    add x14, x13, x7
    csel x14, x13, x14, ls

    add x15, x14, x7
    cmp x0, #8
    csel x15, x14, x15, ne

    cmp x1, #8
    bne 4f

    st1 {v8.d}[0], [x6]
    st1 {v10.d}[0], [x9]
    st1 {v12.d}[0], [x10]
    st1 {v14.d}[0], [x11]
    st1 {v16.d}[0], [x12]
    st1 {v18.d}[0], [x13]
    st1 {v20.d}[0], [x14]
    st1 {v22.d}[0], [x15]

    sub sp, sp, #128
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
    ret

4:
    cmp x1, #4
    blo 5f

    st1 {v8.s}[0], [x6], #4
    st1 {v10.s}[0], [x9], #4
    st1 {v12.s}[0], [x10], #4
    st1 {v14.s}[0], [x11], #4
    st1 {v16.s}[0], [x12], #4
    st1 {v18.s}[0], [x13], #4
    st1 {v20.s}[0], [x14], #4
    st1 {v22.s}[0], [x15], #4

    sub x1, x1, #4
    ext v8.8b, v8.8b, v8.8b, #4
    ext v10.8b, v10.8b, v10.8b, #4
    ext v12.8b, v12.8b, v12.8b, #4
    ext v14.8b, v14.8b, v14.8b, #4
    ext v16.8b, v16.8b, v16.8b, #4
    ext v18.8b, v18.8b, v18.8b, #4
    ext v20.8b, v20.8b, v20.8b, #4
    ext v22.8b, v22.8b, v22.8b, #4

5:
    cmp x1, #2
    blo 6f

    st1 {v8.h}[0], [x6], #2
    st1 {v10.h}[0], [x9], #2
    st1 {v12.h}[0], [x10], #2
    st1 {v14.h}[0], [x11], #2
    st1 {v16.h}[0], [x12], #2
    st1 {v18.h}[0], [x13], #2
    st1 {v20.h}[0], [x14], #2
    st1 {v22.h}[0], [x15], #2

    sub x1, x1, #2
    ext v8.8b, v8.8b, v8.8b, #2
    ext v10.8b, v10.8b, v10.8b, #2
    ext v12.8b, v12.8b, v12.8b, #2
    ext v14.8b, v14.8b, v14.8b, #2
    ext v16.8b, v16.8b, v16.8b, #2
    ext v18.8b, v18.8b, v18.8b, #2
    ext v20.8b, v20.8b, v20.8b, #2
    ext v22.8b, v22.8b, v22.8b, #2

6:
    cmp x1, #1
    blo 7f

    st1 {v8.b}[0], [x6]
    st1 {v10.b}[0], [x9]
    st1 {v12.b}[0], [x10]
    st1 {v14.b}[0], [x11]
    st1 {v16.b}[0], [x12]
    st1 {v18.b}[0], [x13]
    st1 {v20.b}[0], [x14]
    st1 {v22.b}[0], [x15]


7:
sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
