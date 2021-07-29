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

asm_function GEMM_FP16_N8
//void GEMM_FP16_N8(float* dst, const float* src, const float* weight, int src_depth,
//                            int dst_step, int dst_depth, int width, float *bias, int64_t relu)
//Auto Load:
//x0:dst, x1:src, x2:weight, x3:src_depth, x4:dst_step, x5:dst_depth, x6: width, x7: bias

.macro COMPUTE_UNIT_0 z0 z1 z2 z3 z4 z5 z6 z7 y
fmla \z0, \y, v0.h[0]
fmla \z1, \y, v0.h[1]
fmla \z2, \y, v0.h[2]
fmla \z3, \y, v0.h[3]
fmla \z4, \y, v0.h[4]
fmla \z5, \y, v0.h[5]
fmla \z6, \y, v0.h[6]
fmla \z7, \y, v0.h[7]
.endm

.macro COMPUTE_UNIT_1 z0 z1 z2 z3 z4 z5 z6 z7 y
fmla \z0, \y, v1.h[0]
fmla \z1, \y, v1.h[1]
fmla \z2, \y, v1.h[2]
fmla \z3, \y, v1.h[3]
fmla \z4, \y, v1.h[4]
fmla \z5, \y, v1.h[5]
fmla \z6, \y, v1.h[6]
fmla \z7, \y, v1.h[7]
.endm

.macro COMPUTE_UNIT_2 z0 z1 z2 z3 z4 z5 z6 z7 y
fmla \z0, \y, v2.h[0]
fmla \z1, \y, v2.h[1]
fmla \z2, \y, v2.h[2]
fmla \z3, \y, v2.h[3]
fmla \z4, \y, v2.h[4]
fmla \z5, \y, v2.h[5]
fmla \z6, \y, v2.h[6]
fmla \z7, \y, v2.h[7]
.endm

.macro COMPUTE_UNIT_3 z0 z1 z2 z3 z4 z5 z6 z7 y
fmla \z0, \y, v3.h[0]
fmla \z1, \y, v3.h[1]
fmla \z2, \y, v3.h[2]
fmla \z3, \y, v3.h[3]
fmla \z4, \y, v3.h[4]
fmla \z5, \y, v3.h[5]
fmla \z6, \y, v3.h[6]
fmla \z7, \y, v3.h[7]
.endm

.macro COMPUTE_M4_UNIT_0 z0 z1 z2 z3 y
fmla \z0, \y, v0.h[0]
fmla \z1, \y, v0.h[1]
fmla \z2, \y, v0.h[2]
fmla \z3, \y, v0.h[3]
.endm

.macro COMPUTE_M4_UNIT_1 z0 z1 z2 z3 y
fmla \z0, \y, v1.h[0]
fmla \z1, \y, v1.h[1]
fmla \z2, \y, v1.h[2]
fmla \z3, \y, v1.h[3]
.endm

.macro COMPUTE_M4_UNIT_2 z0 z1 z2 z3 y
fmla \z0, \y, v2.h[0]
fmla \z1, \y, v2.h[1]
fmla \z2, \y, v2.h[2]
fmla \z3, \y, v2.h[3]
.endm

.macro COMPUTE_M4_UNIT_3 z0 z1 z2 z3 y
fmla \z0, \y, v3.h[0]
fmla \z1, \y, v3.h[1]
fmla \z2, \y, v3.h[2]
fmla \z3, \y, v3.h[3]
.endm

.macro COMPUTE_M3_UNIT_0 z0 z1 z2 y
fmla \z0, \y, v0.h[0]
fmla \z1, \y, v0.h[1]
fmla \z2, \y, v0.h[2]
.endm

.macro COMPUTE_M3_UNIT_1 z0 z1 z2 y
fmla \z0, \y, v1.h[0]
fmla \z1, \y, v1.h[1]
fmla \z2, \y, v1.h[2]
.endm

.macro COMPUTE_M3_UNIT_2 z0 z1 z2 y
fmla \z0, \y, v2.h[0]
fmla \z1, \y, v2.h[1]
fmla \z2, \y, v2.h[2]
.endm

.macro COMPUTE_M3_UNIT_3 z0 z1 z2 y
fmla \z0, \y, v3.h[0]
fmla \z1, \y, v3.h[1]
fmla \z2, \y, v3.h[2]
.endm

//step multi by sizeof(__fp16)
lsl x4, x4, #1

prfm pldl1keep, [x7]
sub sp, sp, #160
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
stp x19, x20, [sp], #16
stp x21, x22, [sp], #16

//x8: src_z_step (width * 8 * sizeof(__fp16))
lsl x8, x6, #4

//x9: weight_z_step (src_depth * 8 * sizeof(__fp16))
lsl x9, x3, #4
mov x19, x6
// src_depth aligned with 8
lsr x13, x3, #3
lsl x13, x13, #3
// src_depth remain
sub x20, x3, x13
mov x3, x13
mov x21, x20

// x22: Store Flag
ldr x22, [sp, #0]

// src ptr tmp
mov x14, x1

LoopDz:
// load bias
ld1 {v31.16b}, [x7], #16
// dst ptr tmp
mov x10, x0
// weight ptr tmp
mov x15, x2

L16:
    cmp x6, #15
    ble L8

    mov x12, x2
    ldr q0, [x1, #0]
    prfm pldl1keep, [x1, #1024]
    mov v8.16b,  v31.16b
    mov v9.16b,  v31.16b
    ldr q1, [x1, #16]
    mov v10.16b, v31.16b
    mov v11.16b, v31.16b
    ldr q4, [x2, #0]
    prfm pldl1keep, [x2, #512]
    mov v12.16b, v31.16b
    mov v13.16b, v31.16b
    mov v14.16b, v31.16b
    mov v15.16b, v31.16b
    mov v16.16b, v31.16b
    mov v17.16b, v31.16b
    mov v18.16b, v31.16b
    mov v19.16b, v31.16b
    mov v20.16b, v31.16b
    mov v21.16b, v31.16b
    mov v22.16b, v31.16b
    mov v23.16b, v31.16b

    cmp x13, #7
    ble L16MAC8CEND

L16MAC8C:
    ldr q2, [x1, #32]
    ldr q3, [x1, #48]
    ldr q5, [x2, #16]
    ldr q6, [x2, #32]
    // oc8ic0 * [ic0m0, ic0m7]
    COMPUTE_UNIT_0 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v4.8h

    ldr q0, [x1, #64]
    // oc8ic0 * [ic0m8, ic0m15]
    COMPUTE_UNIT_1 v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h, v4.8h

    ldr q1, [x1, #80]
    ldr q7, [x2, #48]
    // oc8ic1 * [ic1m0, ic1m7]
    COMPUTE_UNIT_2 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v5.8h

    ldr q2, [x1, #96]
    ldr q4, [x2, #64]
    // oc8ic1 * [ic1m8, ic1m15]
    COMPUTE_UNIT_3 v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h, v5.8h

    ldr q3, [x1, #112]
    ldr q5, [x2, #80]
    // oc8ic2 * [ic2m0, ic2m7]
    COMPUTE_UNIT_0 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v6.8h

    ldr q0, [x1, #128]
    // oc8ic2 * [ic2m8, ic2m15]
    COMPUTE_UNIT_1 v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h, v6.8h

    ldr q1, [x1, #144]
    ldr q6, [x2, #96]
    // oc8ic3 * [ic3m0, ic3m7]
    COMPUTE_UNIT_2 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v7.8h

    ldr q2, [x1, #160]
    // oc8ic3 * [ic3m8, ic3m15]
    COMPUTE_UNIT_3 v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h, v7.8h

    ldr q3, [x1, #176]
    ldr q7, [x2, #112]
    add x2, x2, #128
    // oc8ic4 * [ic4m0, ic4m7]
    COMPUTE_UNIT_0 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v4.8h

    ldr q0, [x1, #192]
    // oc8ic4 * [ic4m8, ic4m15]
    COMPUTE_UNIT_1 v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h, v4.8h

    ldr q1, [x1, #208]
    ldr q4, [x2, #0]
    prfm pldl1keep, [x2, #512]    
    // oc8ic5 * [ic5m0, ic5m7]
    COMPUTE_UNIT_2 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v5.8h

    ldr q2, [x1, #224]
    // oc8ic5 * [ic5m8, ic5m15]
    COMPUTE_UNIT_3 v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h, v5.8h

    ldr q3, [x1, #240]
    add x1, x1, #256
    // oc8ic6 * [ic6m0, ic6m7]
    COMPUTE_UNIT_0 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v6.8h

    ldr q0, [x1, #0]
    prfm pldl1keep, [x1, #1024]
    // oc8ic6 * [ic6m8, ic6m15]
    COMPUTE_UNIT_1 v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h, v6.8h

    ldr q1, [x1, #16]
    // oc8ic7 * [ic7m0, ic7m7]
    COMPUTE_UNIT_2 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v7.8h

    subs x3, x3, #8
    // oc8ic7 * [ic7m8, ic7m15]
    COMPUTE_UNIT_3 v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h, v7.8h

    bne L16MAC8C

L16MAC8CEND:
    cmp x21, #0
    beq L16MAC1CEND

L16MAC1C:
    // add pre-load offset
    add x1, x1, #32
    add x2, x2, #16
    // oc8ic0 * [ic0m0, ic0m7]
    COMPUTE_UNIT_0 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v4.8h
    ldr q0, [x1, #0]
    subs x20, x20, #1
    // oc8ic0 * [ic0m8, ic0m15]
    COMPUTE_UNIT_1 v16.8h, v17.8h, v18.8h, v19.8h, v20.8h, v21.8h, v22.8h, v23.8h, v4.8h
    ldr q1, [x1, #16]
    ldr q4, [x2]
    bne L16MAC1C

L16MAC1CEND:

    cbz x22, Store16
    eor v0.16b, v0.16b, v0.16b
    fmax v8.8h,  v8.8h,  v0.8h
    fmax v9.8h,  v9.8h,  v0.8h
    fmax v10.8h, v10.8h, v0.8h
    fmax v11.8h, v11.8h, v0.8h
    fmax v12.8h, v12.8h, v0.8h
    fmax v13.8h, v13.8h, v0.8h
    fmax v14.8h, v14.8h, v0.8h
    fmax v15.8h, v15.8h, v0.8h
    fmax v16.8h, v16.8h, v0.8h
    fmax v17.8h, v17.8h, v0.8h
    fmax v18.8h, v18.8h, v0.8h
    fmax v19.8h, v19.8h, v0.8h
    fmax v20.8h, v20.8h, v0.8h
    fmax v21.8h, v21.8h, v0.8h
    fmax v22.8h, v22.8h, v0.8h
    fmax v23.8h, v23.8h, v0.8h

    // if relu6
    cmp x22, #2
    bne Store16
    // 6.0f
    movi v1.8h, #0x46, lsl #8
    fmin v8.8h,  v8.8h,  v1.8h
    fmin v9.8h,  v9.8h,  v1.8h
    fmin v10.8h, v10.8h, v1.8h
    fmin v11.8h, v11.8h, v1.8h
    fmin v12.8h, v12.8h, v1.8h
    fmin v13.8h, v13.8h, v1.8h
    fmin v14.8h, v14.8h, v1.8h
    fmin v15.8h, v15.8h, v1.8h
    fmin v16.8h, v16.8h, v1.8h
    fmin v17.8h, v17.8h, v1.8h
    fmin v18.8h, v18.8h, v1.8h
    fmin v19.8h, v19.8h, v1.8h
    fmin v20.8h, v20.8h, v1.8h
    fmin v21.8h, v21.8h, v1.8h
    fmin v22.8h, v22.8h, v1.8h
    fmin v23.8h, v23.8h, v1.8h
Store16:
    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
    sub x6, x6, #16
    st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], #64
    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
    cmp x6, #16
    // reset weight ptr
    mov x2, x12
    // reset loop counter
    mov x3, x13
    mov x20, x21
    bge L16

// corner case
L8:
    cmp x6, #7
    ble L4

    mov x12, x2
    ldr q0, [x1, #0]
    prfm pldl1keep, [x1, #512]
    mov v8.16b,  v31.16b
    mov v9.16b,  v31.16b
    mov v10.16b, v31.16b
    mov v11.16b, v31.16b
    ldr q4, [x2, #0]
    prfm pldl1keep, [x2, #512]
    mov v12.16b, v31.16b
    mov v13.16b, v31.16b
    mov v14.16b, v31.16b
    mov v15.16b, v31.16b

    cmp x13, #7
    ble L8MAC8CEND

L8MAC8C:
    ldr q1, [x1, #16]
    ldr q2, [x1, #32]
    ldr q5, [x2, #16]
    ldr q6, [x2, #32]
    // oc8ic0 * [ic0m0, ic0m7]
    COMPUTE_UNIT_0 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v4.8h

    ldr q3, [x1, #48]
    ldr q0, [x1, #64]
    ldr q7, [x2, #48]
    ldr q4, [x2, #64]
    // oc8ic1 * [ic1m0, ic1m7]
    COMPUTE_UNIT_1 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v5.8h

    ldr q1, [x1, #80]
    ldr q5, [x2, #80]
    // oc8ic2 * [ic2m0, ic2m7]
    COMPUTE_UNIT_2 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v6.8h

    ldr q2, [x1, #96]
    ldr q6, [x2, #96]
    // oc8ic3 * [ic3m0, ic3m7]
    COMPUTE_UNIT_3 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v7.8h

    ldr q3, [x1, #112]
    ldr q7, [x2, #112]
    // add x1, x1, x8
    add x1, x1, #128
    add x2, x2, #128
    // oc8ic4 * [ic4m0, ic4m7]
    COMPUTE_UNIT_0 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v4.8h

    ldr q0, [x1, #0]
    prfm pldl1keep, [x1, #512]
    // oc8ic5 * [ic5m0, ic5m7]
    COMPUTE_UNIT_1 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v5.8h

    ldr q4, [x2, #0]
    prfm pldl1keep, [x2, #512]
    // oc8ic6 * [ic6m0, ic6m7]
    COMPUTE_UNIT_2 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v6.8h

    subs x3, x3, #8
    // oc8ic7 * [ic7m0, ic7m7]
    COMPUTE_UNIT_3 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v7.8h

    bne L8MAC8C

L8MAC8CEND:
    cmp x21, #0
    beq L8MAC1CEND

L8MAC1C:
    // add pre-load offset
    add x1, x1, #16
    add x2, x2, #16
    subs x20, x20, #1
    // oc8ic0 * [ic0m0, ic0m7]
    COMPUTE_UNIT_0 v8.8h,  v9.8h,  v10.8h, v11.8h, v12.8h, v13.8h, v14.8h, v15.8h, v4.8h
    ldr q0, [x1]
    ldr q4, [x2]
    bne L8MAC1C

L8MAC1CEND:
    cbz x22, Store8
    eor v0.16b, v0.16b, v0.16b
    fmax v8.8h,  v8.8h,  v0.8h
    fmax v9.8h,  v9.8h,  v0.8h
    fmax v10.8h, v10.8h, v0.8h
    fmax v11.8h, v11.8h, v0.8h
    fmax v12.8h, v12.8h, v0.8h
    fmax v13.8h, v13.8h, v0.8h
    fmax v14.8h, v14.8h, v0.8h
    fmax v15.8h, v15.8h, v0.8h

    // if relu6
    cmp x22, #2
    bne Store8
    // 6.0f
    movi v1.8h, #0x46, lsl #8
    fmin v8.8h,  v8.8h,  v1.8h
    fmin v9.8h,  v9.8h,  v1.8h
    fmin v10.8h, v10.8h, v1.8h
    fmin v11.8h, v11.8h, v1.8h
    fmin v12.8h, v12.8h, v1.8h
    fmin v13.8h, v13.8h, v1.8h
    fmin v14.8h, v14.8h, v1.8h
    fmin v15.8h, v15.8h, v1.8h
Store8:
    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
    sub x6, x6, #8
    st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64

    cmp x6, #8

    mov x2, x12
    mov x3, x13
    mov x20, x21
    bge L8

L4:
    cmp x6, #3
    ble L1

    mov x12, x2
    ldr d0, [x1, #0]
    ldr q4, [x2, #0]
    prfm pldl1keep, [x1, #256]
    prfm pldl1keep, [x2, #512]
    mov v8.16b,  v31.16b
    mov v9.16b,  v31.16b
    mov v10.16b, v31.16b
    mov v11.16b, v31.16b

    cmp x13, #7
    ble L4MAC8CEND

L4MAC8C:
    ldr d1, [x1, #8]
    ldr d2, [x1, #16]
    ldr q5, [x2, #16]
    ldr q6, [x2, #32]
    // oc8ic0 * [ic0m0, ic0m3]
    COMPUTE_M4_UNIT_0 v8.8h,  v9.8h,  v10.8h, v11.8h, v4.8h

    ldr d3, [x1, #24]
    ldr d0, [x1, #32]
    ldr q7, [x2, #48]
    ldr q4, [x2, #64]
    // oc8ic1 * [ic1m0, ic1m3]
    COMPUTE_M4_UNIT_1 v8.8h,  v9.8h,  v10.8h, v11.8h, v5.8h

    ldr d1, [x1, #40]
    ldr q5, [x2, #80]
    // oc8ic2 * [ic2m0, ic2m3]
    COMPUTE_M4_UNIT_2 v8.8h,  v9.8h,  v10.8h, v11.8h, v6.8h

    ldr d2, [x1, #48]
    ldr q6, [x2, #96]
    // oc8ic3 * [ic3m0, ic3m3]
    COMPUTE_M4_UNIT_3 v8.8h,  v9.8h,  v10.8h, v11.8h, v7.8h

    ldr d3, [x1, #56]
    ldr q7, [x2, #112]
    // add x1, x1, x8
    add x1, x1, #64
    add x2, x2, #128
    // oc8ic4 * [ic4m0, ic4m3]
    COMPUTE_M4_UNIT_0 v8.8h,  v9.8h,  v10.8h, v11.8h, v4.8h

    ldr d0, [x1, #0]
    prfm pldl1keep, [x1, #256]
    // oc8ic5 * [ic5m0, ic5m3]
    COMPUTE_M4_UNIT_1 v8.8h,  v9.8h,  v10.8h, v11.8h, v5.8h

    ldr q4, [x2, #0]
    prfm pldl1keep, [x2, #512]
    // oc8ic6 * [ic6m0, ic6m3]
    COMPUTE_M4_UNIT_2 v8.8h,  v9.8h,  v10.8h, v11.8h, v6.8h

    subs x3, x3, #8
    // oc8ic7 * [ic7m0, ic7m3]
    COMPUTE_M4_UNIT_3 v8.8h,  v9.8h,  v10.8h, v11.8h, v7.8h

    bne L4MAC8C

L4MAC8CEND:
    cmp x21, #0
    beq L4MAC1CEND

L4MAC1C:
    // add pre-load offset
    add x1, x1, #8
    add x2, x2, #16
    subs x20, x20, #1
    // oc8ic0 * [ic0m0, ic0m3]
    COMPUTE_M4_UNIT_0 v8.8h,  v9.8h,  v10.8h, v11.8h, v4.8h
    ldr q0, [x1]
    ldr q4, [x2]
    bne L4MAC1C

L4MAC1CEND:
    cbz x22, Store4
    eor v0.16b, v0.16b, v0.16b
    fmax v8.8h,  v8.8h,  v0.8h
    fmax v9.8h,  v9.8h,  v0.8h
    fmax v10.8h, v10.8h, v0.8h
    fmax v11.8h, v11.8h, v0.8h

    // if relu6
    cmp x22, #2
    bne Store4
    // 6.0f
    movi v1.8h, #0x46, lsl #8
    fmin v8.8h,  v8.8h,  v1.8h
    fmin v9.8h,  v9.8h,  v1.8h
    fmin v10.8h, v10.8h, v1.8h
    fmin v11.8h, v11.8h, v1.8h
Store4:
    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
    sub x6, x6, #4

    cmp x6, #4

    mov x2, x12
    mov x3, x13
    mov x20, x21
    bge L4

L1:
    cmp x6, #0
    ble END

    // when L1, src is 4 x crr
    mov x12, x2
    ldr d0, [x1, #0]
    ldr q4, [x2, #0]
    prfm pldl1keep, [x1, #256]
    prfm pldl1keep, [x2, #512]
    mov v8.16b,  v31.16b
    mov v9.16b,  v31.16b
    mov v10.16b, v31.16b

    cmp x13, #7
    ble L1MAC8CEND

L1MAC8C:
    ldr d1, [x1, #8]
    ldr d2, [x1, #16]
    ldr q5, [x2, #16]
    ldr q6, [x2, #32]
    // oc8ic0 * [ic0m0, ic0m2]
    COMPUTE_M3_UNIT_0 v8.8h,  v9.8h,  v10.8h, v4.8h

    ldr d3, [x1, #24]
    ldr d0, [x1, #32]
    ldr q7, [x2, #48]
    ldr q4, [x2, #64]
    // oc8ic1 * [ic1m0, ic1m2]
    COMPUTE_M3_UNIT_1 v8.8h,  v9.8h,  v10.8h, v5.8h

    ldr d1, [x1, #40]
    ldr q5, [x2, #80]
    // oc8ic2 * [ic2m0, ic2m2]
    COMPUTE_M3_UNIT_2 v8.8h,  v9.8h,  v10.8h, v6.8h

    ldr d2, [x1, #48]
    ldr q6, [x2, #96]
    // oc8ic3 * [ic3m0, ic3m2]
    COMPUTE_M3_UNIT_3 v8.8h,  v9.8h,  v10.8h, v7.8h

    ldr d3, [x1, #56]
    ldr q7, [x2, #112]
    // add x1, x1, x8
    add x1, x1, #64
    add x2, x2, #128
    // oc8ic4 * [ic4m0, ic4m2]
    COMPUTE_M3_UNIT_0 v8.8h,  v9.8h,  v10.8h, v4.8h

    ldr d0, [x1, #0]
    prfm pldl1keep, [x1, #256]
    // oc8ic5 * [ic5m0, ic5m2]
    COMPUTE_M3_UNIT_1 v8.8h,  v9.8h,  v10.8h, v5.8h

    ldr q4, [x2, #0]
    prfm pldl1keep, [x2, #512]
    // oc8ic6 * [ic6m0, ic6m2]
    COMPUTE_M3_UNIT_2 v8.8h,  v9.8h,  v10.8h, v6.8h

    subs x3, x3, #8
    // oc8ic7 * [ic7m0, ic7m2]
    COMPUTE_M3_UNIT_3 v8.8h,  v9.8h,  v10.8h, v7.8h

    bne L1MAC8C

L1MAC8CEND:
    cmp x21, #0
    beq L1MAC1CEND

L1MAC1C:
    // add pre-load offset
    add x1, x1, #8
    add x2, x2, #16
    subs x20, x20, #1
    // oc8ic0 * [ic0m0, ic0m2]
    COMPUTE_M3_UNIT_0 v8.8h,  v9.8h,  v10.8h, v4.8h
    ldr q0, [x1]
    ldr q4, [x2]
    bne L1MAC1C

L1MAC1CEND:
    cbz x22, Store1
    eor v0.16b, v0.16b, v0.16b
    fmax v8.8h,  v8.8h,  v0.8h
    fmax v9.8h,  v9.8h,  v0.8h
    fmax v10.8h, v10.8h, v0.8h

    // if relu6
    cmp x22, #2
    bne Store1
    // 6.0f
    movi v1.8h, #0x46, lsl #8
    fmin v8.8h,  v8.8h,  v1.8h
    fmin v9.8h,  v9.8h,  v1.8h
    fmin v10.8h, v10.8h, v1.8h
Store1:
    cmp x6, #3
    beq Store1_3reg
    cmp x6, #2
    beq Store1_2reg
    cmp x6, #1
    beq Store1_1reg
Store1_3reg:
    st1 {v8.8h, v9.8h, v10.8h}, [x0]
    b Store1_End
Store1_2reg:
    st1 {v8.8h, v9.8h}, [x0]
    b Store1_End
Store1_1reg:
    st1 {v8.8h}, [x0]
Store1_End:
    // reset ptr and counters
    mov x2, x12
    mov x3, x13
    mov x20, x21

END:

subs x5, x5, #8
// update dst ptr
add x0, x10, x4
// reset src ptr
mov x1, x14
// update weight ptr
add x2, x15, x9
// reset M counter
mov x6, x19
bne LoopDz

sub sp, sp, #160
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ldp x19, x20, [sp], #16
ldp x21, x22, [sp], #16

ret

#endif
#endif
