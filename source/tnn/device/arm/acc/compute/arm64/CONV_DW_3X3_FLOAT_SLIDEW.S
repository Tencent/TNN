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

asm_function ConvDw3x3FloatSlideW 
//void ConvDw3x3FloatSlideW(float *dst_z,
//                        float **cache_line,
//                        const float* weight_z,
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
w_22      .req v16


//Auto Load:
//x0:dst_z, x1:cache_line, x2:weight_z, x3: dst_width

cmp width, #0
ble End

ldr x4, [x1]
ldr x5, [x1, #8]
ldr x6, [x1, #16]

ld1 {w_00.4s, w_01.4s, w_02.4s}, [weight], #48
ld1 {w_10.4s, w_11.4s, w_12.4s}, [weight], #48
ld1 {w_20.4s, w_21.4s}, [weight], #32
ld1 {w_22.4s}, [weight]

ld1 {v21.4s}, [line0], #16
ld1 {v22.4s}, [line1], #16
ld1 {v23.4s}, [line2], #16

fmul v17.4s, v21.4s, w_00.4s
fmla v17.4s, v22.4s, w_10.4s
fmla v17.4s, v23.4s, w_20.4s

ld1 {v21.4s}, [line0], #16
ld1 {v22.4s}, [line1], #16
ld1 {v23.4s}, [line2], #16

fmul v18.4s, v21.4s, w_00.4s
fmla v17.4s, v21.4s, w_01.4s
fmla v18.4s, v22.4s, w_10.4s
fmla v17.4s, v22.4s, w_11.4s
fmla v18.4s, v23.4s, w_20.4s
fmla v17.4s, v23.4s, w_21.4s

subs width, width, #1
beq LoopDwEnd
LoopDw:
    ld1 {v21.4s}, [line0], #16
    ld1 {v22.4s}, [line1], #16
    ld1 {v23.4s}, [line2], #16

    fmul v19.4s, v21.4s, w_00.4s
    fmla v18.4s, v21.4s, w_01.4s
    fmla v17.4s, v21.4s, w_02.4s

    fmla v19.4s, v22.4s, w_10.4s
    fmla v18.4s, v22.4s, w_11.4s
    fmla v17.4s, v22.4s, w_12.4s

    fmla v19.4s, v23.4s, w_20.4s
    fmla v18.4s, v23.4s, w_21.4s
    fmla v17.4s, v23.4s, w_22.4s

    st1 {v17.4s}, [dst], #16
    subs width, width, #1
    mov v17.16b, v18.16b
    mov v18.16b, v19.16b

    bne LoopDw
LoopDwEnd:
ld1 {v21.4s}, [line0], #16
ld1 {v22.4s}, [line1], #16
ld1 {v23.4s}, [line2], #16
fmla v17.4s, v21.4s, w_02.4s
fmla v17.4s, v22.4s, w_12.4s
fmla v17.4s, v23.4s, w_22.4s
st1 {v17.4s}, [dst], #16

End:

ret

#endif
