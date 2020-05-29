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

asm_function ConvDw5x5FloatSlideW 
//void ConvDw5x5FloatSlideW(float *dst_z,
//                        float **cache_line,
//                        const float* weight_z,
//                        int dst_width)

dst      .req x0
line0    .req x4
line1    .req x5
line2    .req x6
line3    .req x7
line4    .req x8
weight   .req x2
width    .req x3

w_00      .req v0
w_01      .req v1
w_02      .req v2
w_03      .req v3
w_04      .req v4
w_10      .req v5
w_11      .req v6
w_12      .req v7
w_13      .req v8
w_14      .req v9
w_20      .req v10
w_21      .req v11
w_22      .req v12
w_23      .req v13
w_24      .req v14
w_30      .req v15
w_31      .req v16
w_32      .req v17
w_33      .req v18
w_34      .req v19
w_40      .req v20
w_41      .req v21
w_42      .req v22
w_43      .req v23
w_44      .req v24

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
//Auto Load:
//x0:dst_z, x1:cache_line, x2:weight_z, x3: dst_width

cmp width, #0
ble End

ldr x4, [x1]
ldr x5, [x1, #8]
ldr x6, [x1, #16]
ldr x7, [x1, #24]
ldr x8, [x1, #32]

ld1 {w_00.4s, w_01.4s, w_02.4s, w_03.4s}, [weight], #64
ld1 {w_04.4s, w_10.4s, w_11.4s, w_12.4s}, [weight], #64
ld1 {w_13.4s, w_14.4s, w_20.4s, w_21.4s}, [weight], #64
ld1 {w_22.4s, w_23.4s, w_24.4s, w_30.4s}, [weight], #64
ld1 {w_31.4s, w_32.4s, w_33.4s, w_34.4s}, [weight], #64
ld1 {w_40.4s, w_41.4s, w_42.4s, w_43.4s}, [weight], #64
ld1 {w_44.4s}, [weight]

ld1 {v30.4s}, [line0], #16
ld1 {v31.4s}, [line1], #16
fmul v25.4s, v30.4s, w_00.4s
fmla v25.4s, v31.4s, w_10.4s
ld1 {v30.4s}, [line2], #16
ld1 {v31.4s}, [line3], #16
fmla v25.4s, v30.4s, w_20.4s
fmla v25.4s, v31.4s, w_30.4s
ld1 {v30.4s}, [line4], #16
fmla v25.4s, v30.4s, w_40.4s

ld1 {v30.4s}, [line0], #16
ld1 {v31.4s}, [line1], #16
fmul v26.4s, v30.4s, w_00.4s
fmla v25.4s, v30.4s, w_01.4s
fmla v26.4s, v31.4s, w_10.4s
fmla v25.4s, v31.4s, w_11.4s
ld1 {v30.4s}, [line2], #16
ld1 {v31.4s}, [line3], #16
fmla v26.4s, v30.4s, w_20.4s
fmla v25.4s, v30.4s, w_21.4s
fmla v26.4s, v31.4s, w_30.4s
fmla v25.4s, v31.4s, w_31.4s
ld1 {v30.4s}, [line4], #16
fmla v26.4s, v30.4s, w_40.4s
fmla v25.4s, v30.4s, w_41.4s

ld1 {v30.4s}, [line0], #16
ld1 {v31.4s}, [line1], #16
fmul v27.4s, v30.4s, w_00.4s
fmla v26.4s, v30.4s, w_01.4s
fmla v25.4s, v30.4s, w_02.4s
fmla v27.4s, v31.4s, w_10.4s
fmla v26.4s, v31.4s, w_11.4s
fmla v25.4s, v31.4s, w_12.4s
ld1 {v30.4s}, [line2], #16
ld1 {v31.4s}, [line3], #16
fmla v27.4s, v30.4s, w_20.4s
fmla v26.4s, v30.4s, w_21.4s
fmla v25.4s, v30.4s, w_22.4s
fmla v27.4s, v31.4s, w_30.4s
fmla v26.4s, v31.4s, w_31.4s
fmla v25.4s, v31.4s, w_32.4s
ld1 {v30.4s}, [line4], #16
fmla v27.4s, v30.4s, w_40.4s
fmla v26.4s, v30.4s, w_41.4s
fmla v25.4s, v30.4s, w_42.4s

ld1 {v30.4s}, [line0], #16
ld1 {v31.4s}, [line1], #16
fmul v28.4s, v30.4s, w_00.4s
fmla v27.4s, v30.4s, w_01.4s
fmla v26.4s, v30.4s, w_02.4s
fmla v25.4s, v30.4s, w_03.4s
fmla v28.4s, v31.4s, w_10.4s
fmla v27.4s, v31.4s, w_11.4s
fmla v26.4s, v31.4s, w_12.4s
fmla v25.4s, v31.4s, w_13.4s
ld1 {v30.4s}, [line2], #16
ld1 {v31.4s}, [line3], #16
fmla v28.4s, v30.4s, w_20.4s
fmla v27.4s, v30.4s, w_21.4s
fmla v26.4s, v30.4s, w_22.4s
fmla v25.4s, v30.4s, w_23.4s
fmla v28.4s, v31.4s, w_30.4s
fmla v27.4s, v31.4s, w_31.4s
fmla v26.4s, v31.4s, w_32.4s
fmla v25.4s, v31.4s, w_33.4s
ld1 {v30.4s}, [line4], #16
fmla v28.4s, v30.4s, w_40.4s
fmla v27.4s, v30.4s, w_41.4s
fmla v26.4s, v30.4s, w_42.4s
fmla v25.4s, v30.4s, w_43.4s

subs width, width, #1
beq LoopDwEnd
LoopDw:
    ld1 {v30.4s}, [line0], #16
    ld1 {v31.4s}, [line1], #16
    fmul v29.4s, v30.4s, w_00.4s
    fmla v28.4s, v30.4s, w_01.4s
    fmla v27.4s, v30.4s, w_02.4s
    fmla v26.4s, v30.4s, w_03.4s
    fmla v25.4s, v30.4s, w_04.4s

    fmla v29.4s, v31.4s, w_10.4s
    fmla v28.4s, v31.4s, w_11.4s
    ld1 {v30.4s}, [line2], #16
    fmla v27.4s, v31.4s, w_12.4s
    fmla v26.4s, v31.4s, w_13.4s
    fmla v25.4s, v31.4s, w_14.4s

    fmla v29.4s, v30.4s, w_20.4s
    fmla v28.4s, v30.4s, w_21.4s
    ld1 {v31.4s}, [line3], #16
    fmla v27.4s, v30.4s, w_22.4s
    fmla v26.4s, v30.4s, w_23.4s
    fmla v25.4s, v30.4s, w_24.4s

    fmla v29.4s, v31.4s, w_30.4s
    fmla v28.4s, v31.4s, w_31.4s
    ld1 {v30.4s}, [line4], #16
    fmla v27.4s, v31.4s, w_32.4s
    fmla v26.4s, v31.4s, w_33.4s
    fmla v25.4s, v31.4s, w_34.4s

    fmla v29.4s, v30.4s, w_40.4s
    fmla v28.4s, v30.4s, w_41.4s
    fmla v27.4s, v30.4s, w_42.4s
    fmla v26.4s, v30.4s, w_43.4s
    fmla v25.4s, v30.4s, w_44.4s
    
    st1 {v25.4s}, [dst], #16
    subs width, width, #1
    mov v25.16b, v26.16b
    mov v26.16b, v27.16b
    mov v27.16b, v28.16b
    mov v28.16b, v29.16b

    bne LoopDw
LoopDwEnd:
ld1 {v30.4s}, [line0], #16
ld1 {v31.4s}, [line1], #16
fmla v25.4s, v30.4s, w_04.4s
fmla v25.4s, v31.4s, w_14.4s
ld1 {v30.4s}, [line2], #16
ld1 {v31.4s}, [line3], #16
fmla v25.4s, v30.4s, w_24.4s
fmla v25.4s, v31.4s, w_34.4s
ld1 {v30.4s}, [line4], #16
fmla v25.4s, v30.4s, w_44.4s
st1 {v25.4s}, [dst], #16

End:

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
