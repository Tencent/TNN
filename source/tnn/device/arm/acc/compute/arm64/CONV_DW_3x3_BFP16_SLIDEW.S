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

asm_function ConvDw3x3Bfp16SlideW 
//void ConvDw3x3Bfp16SlideW(bfp16_t *dst_z,
//                        bfp16_t **cache_line,
//                        const bfp16_t* weight_z,
//                        int dst_width)

dst      .req x0
line0    .req x4
line1    .req x5
line2    .req x6
weight   .req x2
width    .req x3

w_00      .req v0
w_01      .req v1
w_02      .req v2
w_10      .req v3
w_11      .req v4
w_12      .req v5
w_20      .req v6
w_21      .req v7
w_22      .req v8


sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
//Auto Load:
//x0:dst_z, x1:cache_line, x2:weight_z, x3: dst_width

cmp width, #0
ble End

ldr line0, [x1]
ldr line1, [x1, #8]
ldr line2, [x1, #16]

ld1 {w_00.4s, w_01.4s, w_02.4s}, [weight], #48
ld1 {w_10.4s, w_11.4s, w_12.4s}, [weight], #48
ld1 {w_20.4s, w_21.4s, w_22.4s}, [weight], #48

ld1 {v24.4h}, [line0], #8
ld1 {v25.4h}, [line1], #8
ld1 {v26.4h}, [line2], #8
shll v24.4s, v24.4h, #16
shll v25.4s, v25.4h, #16
shll v26.4s, v26.4h, #16

fmul v17.4s, v24.4s, w_00.4s
fmla v17.4s, v25.4s, w_10.4s
fmla v17.4s, v26.4s, w_20.4s

ld1 {v24.4h}, [line0], #8
ld1 {v25.4h}, [line1], #8
ld1 {v26.4h}, [line2], #8
shll v24.4s, v24.4h, #16
shll v25.4s, v25.4h, #16
shll v26.4s, v26.4h, #16

fmul v18.4s, v24.4s, w_00.4s
fmla v17.4s, v24.4s, w_01.4s
fmla v18.4s, v25.4s, w_10.4s
fmla v17.4s, v25.4s, w_11.4s
fmla v18.4s, v26.4s, w_20.4s
fmla v17.4s, v26.4s, w_21.4s

subs width, width, #1

cmp width, #4
blt LoopDw
LoopDwUnroll4:
    
    ld1 {v24.4h,v25.4h, v26.4h, v27.4h}, [line0], #32
    shll v24.4s, v24.4h, #16
    shll v25.4s, v25.4h, #16
    shll v26.4s, v26.4h, #16
    shll v27.4s, v27.4h, #16
    ld1 {v12.4h,v13.4h, v14.4h, v15.4h}, [line1], #32
    shll v12.4s, v12.4h, #16
    shll v13.4s, v13.4h, #16
    shll v14.4s, v14.4h, #16
    shll v15.4s, v15.4h, #16
    ld1 {v28.4h,v29.4h, v30.4h, v31.4h}, [line2], #32
    shll v28.4s, v28.4h, #16
    shll v29.4s, v29.4h, #16
    shll v30.4s, v30.4h, #16
    shll v31.4s, v31.4h, #16

    fmla v17.4s, v24.4s, w_02.4s
    fmla v18.4s, v24.4s, w_01.4s
    fmul v19.4s, v24.4s, w_00.4s


    fmla v17.4s, v12.4s, w_12.4s
    fmla v18.4s, v12.4s, w_11.4s
    fmla v19.4s, v12.4s, w_10.4s

    fmla v17.4s, v28.4s, w_22.4s
    fmla v18.4s, v28.4s, w_21.4s
    fmla v19.4s, v28.4s, w_20.4s

    shrn v17.4h, v17.4s, #16
    fmla v18.4s, v25.4s, w_02.4s
    fmla v19.4s, v25.4s, w_01.4s
    fmul v16.4s, v25.4s, w_00.4s


    fmla v18.4s, v13.4s, w_12.4s
    fmla v19.4s, v13.4s, w_11.4s
    fmla v16.4s, v13.4s, w_10.4s

    st1 {v17.4h}, [dst], #8

    fmla v18.4s, v29.4s, w_22.4s
    fmla v19.4s, v29.4s, w_21.4s
    fmla v16.4s, v29.4s, w_20.4s


    fmla v19.4s, v26.4s, w_02.4s
    fmla v16.4s, v26.4s, w_01.4s
    fmul v17.4s, v26.4s, w_00.4s

    shrn v18.4h, v18.4s, #16
    
    fmla v19.4s, v14.4s, w_12.4s
    fmla v16.4s, v14.4s, w_11.4s
    fmla v17.4s, v14.4s, w_10.4s


    fmla v19.4s, v30.4s, w_22.4s
    fmla v16.4s, v30.4s, w_21.4s
    fmla v17.4s, v30.4s, w_20.4s

    st1 {v18.4h}, [dst], #8

    fmla v16.4s, v27.4s, w_02.4s
    fmla v17.4s, v27.4s, w_01.4s
    fmul v18.4s, v27.4s, w_00.4s

    shrn v19.4h, v19.4s, #16

    fmla v16.4s, v15.4s, w_12.4s
    fmla v17.4s, v15.4s, w_11.4s
    fmla v18.4s, v15.4s, w_10.4s

    st1 {v19.4h}, [dst], #8

    fmla v16.4s, v31.4s, w_22.4s
    fmla v17.4s, v31.4s, w_21.4s
    fmla v18.4s, v31.4s, w_20.4s

    shrn v16.4h, v16.4s, #16

    subs width, width, #4
    cmp width, #4
    st1 {v16.4h}, [dst], #8
    bge LoopDwUnroll4

cmp width, #0
beq LoopDwEnd

LoopDw:
    ld1 {v24.4h}, [line0], #8
    ld1 {v25.4h}, [line1], #8
    ld1 {v26.4h}, [line2], #8
    shll v24.4s, v24.4h, #16
    shll v25.4s, v25.4h, #16
    shll v26.4s, v26.4h, #16

    fmla v17.4s, v24.4s, w_02.4s
    fmla v18.4s, v24.4s, w_01.4s
    fmul v19.4s, v24.4s, w_00.4s

    fmla v17.4s, v25.4s, w_12.4s
    fmla v18.4s, v25.4s, w_11.4s
    fmla v19.4s, v25.4s, w_10.4s

    fmla v17.4s, v26.4s, w_22.4s
    fmla v18.4s, v26.4s, w_21.4s
    fmla v19.4s, v26.4s, w_20.4s

    shrn v31.4h, v17.4s, #16
    subs width, width, #1
    mov v17.16b, v18.16b
    mov v18.16b, v19.16b
    st1 {v31.4h}, [dst], #8

    bne LoopDw
LoopDwEnd:
ld1 {v24.4h}, [line0], #8
ld1 {v25.4h}, [line1], #8
ld1 {v26.4h}, [line2], #8
shll v24.4s, v24.4h, #16
shll v25.4s, v25.4h, #16
shll v26.4s, v26.4h, #16
fmla v17.4s, v24.4s, w_02.4s
fmla v17.4s, v25.4s, w_12.4s
fmla v17.4s, v26.4s, w_22.4s
shrn v31.4h, v17.4s, #16
st1 {v31.4h}, [dst], #8

End:

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
