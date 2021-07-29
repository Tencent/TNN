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

asm_function Half2FloatKernel
//void Half2FloatKernel(float* dst, const __fp16* src, const size_t length)
//Auto Load:
//x0:dst, x1:src, x2:length
cmp x2, #15
ble L16END

L16:
    sub x2, x2, #16
    ldr q0, [x1]
    ldr q1, [x1, #16]
    add x1, x1, #32
    cmp x2, #16
    fcvtl  v2.4s, v0.4h
    fcvtl2 v3.4s, v0.8h
    fcvtl  v4.4s, v1.4h
    fcvtl2 v5.4s, v1.8h
    st1 {v2.4s, v3.4s, v4.4s, v5.4s}, [x0], #64
    bge L16

L16END:
cmp x2, #7
ble L8END
L8:
    sub x2, x2, #8
    ld1 {v0.8h}, [x1], #16
    cmp x2, #8
    fcvtl  v1.4s, v0.4h
    fcvtl2 v2.4s, v0.8h
    st1 {v1.4s, v2.4s}, [x0], #32
    bge L8

L8END:
cmp x2, #3
ble L4END
L4:
    sub x2, x2, #4
    ld1 {v0.4h}, [x1], #8
    cmp x2, #4
    fcvtl  v1.4s, v0.4h
    st1 {v1.4s}, [x0], #16
    bge L4

L4END:
cmp x2, #0
beq L1END

L1:
    subs x2, x2, #1
    ldr h0, [x1], #2
    fcvt s1, h0
    str s1, [x0], #4
    bne L1
L1END:

ret

#endif
#endif
