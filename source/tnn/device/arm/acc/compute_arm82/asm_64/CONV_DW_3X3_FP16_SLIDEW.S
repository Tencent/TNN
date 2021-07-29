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

asm_function ConvDw3x3Fp16SlideW 
//void ConvDw3x3Fp16SlideW(__fp16 *dst_z,
//                        __fp16 **cache_line,
//                        const __fp16* weight_z,
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

ldr line0, [x1]
ldr line1, [x1, #8]
ldr line2, [x1, #16]

ldr q0, [weight, #0]
ldr q1, [weight, #16]
ldr q2, [weight, #32]
ldr q3, [weight, #48]
ldr q4, [weight, #64]
ldr q5, [weight, #80]
ldr q6, [weight, #96]
ldr q7, [weight, #112]
ldr q16, [weight, #128]

ld1 {v21.8h}, [line0], #16
ld1 {v22.8h}, [line1], #16
ld1 {v23.8h}, [line2], #16

fmul v17.8h, v21.8h, w_00.8h
fmla v17.8h, v22.8h, w_10.8h
fmla v17.8h, v23.8h, w_20.8h

ld1 {v21.8h}, [line0], #16
ld1 {v22.8h}, [line1], #16
ld1 {v23.8h}, [line2], #16

fmul v18.8h, v21.8h, w_00.8h
fmla v17.8h, v21.8h, w_01.8h
fmla v18.8h, v22.8h, w_10.8h
fmla v17.8h, v22.8h, w_11.8h
fmla v18.8h, v23.8h, w_20.8h
fmla v17.8h, v23.8h, w_21.8h

subs width, width, #1
beq LoopDwEnd
LoopDw:
    ld1 {v21.8h}, [line0], #16
    ld1 {v22.8h}, [line1], #16
    ld1 {v23.8h}, [line2], #16

    fmul v19.8h, v21.8h, w_00.8h
    fmla v18.8h, v21.8h, w_01.8h
    fmla v17.8h, v21.8h, w_02.8h

    fmla v19.8h, v22.8h, w_10.8h
    fmla v18.8h, v22.8h, w_11.8h
    fmla v17.8h, v22.8h, w_12.8h

    fmla v19.8h, v23.8h, w_20.8h
    fmla v18.8h, v23.8h, w_21.8h
    fmla v17.8h, v23.8h, w_22.8h

    st1 {v17.8h}, [dst], #16
    subs width, width, #1
    mov v17.16b, v18.16b
    mov v18.16b, v19.16b

    bne LoopDw
LoopDwEnd:
ld1 {v21.8h}, [line0], #16
ld1 {v22.8h}, [line1], #16
ld1 {v23.8h}, [line2], #16
fmla v17.8h, v21.8h, w_02.8h
fmla v17.8h, v22.8h, w_12.8h
fmla v17.8h, v23.8h, w_22.8h
st1 {v17.8h}, [dst], #16

End:

ret

#endif
#endif
