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

asm_function ConvBfp16O4
//void ConvBfp16O4(bfp16_t* dst, const bfp16_t* src, const float* weight, int width, int src_w_step, int src_depth_quad, int src_depth_step, int fw, int fh, int dilate_x_step, int dilate_y_step)

//Auto Load:
//x0:dst, x1:src, x2:weight, x3:width, x4:src_w_step, x5:src_depth_quad, x6: src_depth_step, x7:fw

//Load from sp
//x8:fh, x9:dilate_x_step, x10:dilate_y_step
//eor x8, x8, x8
ldr x8, [sp, #0]
//eor x9, x9, x9
ldr x9, [sp, #8]
//eor x10, x10, x10   
ldr x10, [sp, #16]

//step multi by sizeof(bfp16_t)
mov x12, #2
mul x10, x12, x10
mul x9, x12, x9
mul x6, x12, x6
mul x4, x12, x4

//src_depth_step -> src_depth_step - fh*dilate_y_step
mul x12, x8, x10
sub x6, x6, x12

//dilate_y_step -> dilate_y_step-fw*dilate_x_step
mul x12, x7, x9
sub x10, x10, x12

sub sp, sp, #144
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
stp x19, x20, [sp], #16

L14:
cmp x3, #13
ble L8

mov x14, #14
mul x14, x4, x14

L14Loop:
    mov x11, x1
    mov x12, x2
    mov x13, x5
    movi v14.4s, #0
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    movi v24.4s, #0
    movi v25.4s, #0
    movi v26.4s, #0
    movi v27.4s, #0
    L14LoopZ:
        mov x19, x8
        L14LoopFY:
            mov x20, x7
            L14LoopFX:
                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                ld1 {v0.4h}, [x1], x4
                shll v0.4s, v0.4h, #16
                fmla v14.4s, v28.4s, v0.s[0]
                ld1 {v1.4h}, [x1], x4
                shll v1.4s, v1.4h, #16
                fmla v14.4s, v29.4s, v0.s[1]
                fmla v14.4s, v30.4s, v0.s[2]
                ld1 {v2.4h}, [x1], x4
                shll v2.4s, v2.4h, #16
                fmla v15.4s, v28.4s, v1.s[0]
                fmla v14.4s, v31.4s, v0.s[3]
                ld1 {v3.4h}, [x1], x4
                shll v3.4s, v3.4h, #16
                fmla v15.4s, v29.4s, v1.s[1]
                fmla v16.4s, v29.4s, v2.s[1]
                ld1 {v4.4h}, [x1], x4
                shll v4.4s, v4.4h, #16
                fmla v15.4s, v30.4s, v1.s[2]
                fmla v16.4s, v28.4s, v2.s[0]
                fmla v15.4s, v31.4s, v1.s[3]
                ld1 {v5.4h}, [x1], x4
                shll v5.4s, v5.4h, #16
                fmla v16.4s, v30.4s, v2.s[2]
                ld1 {v6.4h}, [x1], x4
                shll v6.4s, v6.4h, #16
                fmla v16.4s, v31.4s, v2.s[3]
                ld1 {v7.4h}, [x1], x4
                shll v7.4s, v7.4h, #16
                fmla v17.4s, v28.4s, v3.s[0]
                fmla v18.4s, v28.4s, v4.s[0]
                fmla v17.4s, v29.4s, v3.s[1]
                ld1 {v8.4h}, [x1], x4
                shll v8.4s, v8.4h, #16
                fmla v17.4s, v30.4s, v3.s[2]
                fmla v18.4s, v29.4s, v4.s[1]
                fmla v17.4s, v31.4s, v3.s[3]

                ld1 {v9.4h}, [x1], x4
                shll v9.4s, v9.4h, #16
                fmla v18.4s, v30.4s, v4.s[2]
                fmla v19.4s, v28.4s, v5.s[0]
                fmla v20.4s, v28.4s, v6.s[0]
                ld1 {v10.4h}, [x1], x4
                shll v10.4s, v10.4h, #16
                fmla v18.4s, v31.4s, v4.s[3]
                fmla v19.4s, v29.4s, v5.s[1]
                fmla v20.4s, v29.4s, v6.s[1]
                ld1 {v11.4h}, [x1], x4
                shll v11.4s, v11.4h, #16
                fmla v19.4s, v30.4s, v5.s[2]
                fmla v20.4s, v30.4s, v6.s[2]
                ld1 {v12.4h}, [x1], x4
                shll v12.4s, v12.4h, #16
                fmla v19.4s, v31.4s, v5.s[3]
                ld1 {v13.4h}, [x1], x4
                shll v13.4s, v13.4h, #16
                fmla v20.4s, v31.4s, v6.s[3]

                fmla v21.4s, v28.4s, v7.s[0]
                fmla v22.4s, v28.4s, v8.s[0]
                fmla v23.4s, v28.4s, v9.s[0]
                fmla v24.4s, v28.4s, v10.s[0]
                fmla v25.4s, v28.4s, v11.s[0]
                fmla v26.4s, v28.4s, v12.s[0]
                fmla v27.4s, v28.4s, v13.s[0]
                
                fmla v21.4s, v29.4s, v7.s[1]
                fmla v22.4s, v29.4s, v8.s[1]
                fmla v23.4s, v29.4s, v9.s[1]
                fmla v24.4s, v29.4s, v10.s[1]
                fmla v25.4s, v29.4s, v11.s[1]
                fmla v26.4s, v29.4s, v12.s[1]
                fmla v27.4s, v29.4s, v13.s[1]

                fmla v21.4s, v30.4s, v7.s[2]
                fmla v22.4s, v30.4s, v8.s[2]
                fmla v23.4s, v30.4s, v9.s[2]
                fmla v24.4s, v30.4s, v10.s[2]
                fmla v25.4s, v30.4s, v11.s[2]
                fmla v26.4s, v30.4s, v12.s[2]
                fmla v27.4s, v30.4s, v13.s[2]

                fmla v21.4s, v31.4s, v7.s[3]
                fmla v22.4s, v31.4s, v8.s[3]
                fmla v23.4s, v31.4s, v9.s[3]
                fmla v24.4s, v31.4s, v10.s[3]
                fmla v25.4s, v31.4s, v11.s[3]
                fmla v26.4s, v31.4s, v12.s[3]
                fmla v27.4s, v31.4s, v13.s[3]

                subs x7, x7, #1
                sub x1, x1, x14

                add x1, x1, x9
                bne L14LoopFX
            subs x8, x8, #1
            mov x7, x20 
            add x1, x1, x10
            bne L14LoopFY
        subs x5, x5, #1
        mov x8, x19
        add x1, x1, x6
        bne L14LoopZ
    sub x3, x3, #14
    shrn v14.4h, v14.4s, #16
    shrn v15.4h, v15.4s, #16
    shrn v16.4h, v16.4s, #16
    shrn v17.4h, v17.4s, #16
    shrn v18.4h, v18.4s, #16
    shrn v19.4h, v19.4s, #16
    shrn v20.4h, v20.4s, #16
    shrn v21.4h, v21.4s, #16
    st1 {v14.4h, v15.4h, v16.4h, v17.4h}, [x0], #32
    add x1, x11, x14
    shrn v22.4h, v22.4s, #16
    shrn v23.4h, v23.4s, #16
    shrn v24.4h, v24.4s, #16
    shrn v25.4h, v25.4s, #16
    st1 {v18.4h, v19.4h, v20.4h, v21.4h}, [x0], #32
    mov x2, x12
    cmp x3, #14
    shrn v26.4h, v26.4s, #16
    shrn v27.4h, v27.4s, #16
    st1 {v22.4h, v23.4h, v24.4h, v25.4h}, [x0], #32
    mov x5, x13
    st1 {v26.4h, v27.4h}, [x0], #16
    bge L14Loop


L8:
cmp x3, #7
ble L4

mov x14, #8
mul x14, x4, x14

L8Loop:
    mov x11, x1
    mov x12, x2
    mov x13, x5
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0
    L8LoopZ:
        mov v27.d[0], x8
        L8LoopFY:
            mov v27.d[1], x7
            L8LoopFX:
                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                ld1 {v16.4s}, [x1], x4
                shll v16.4s, v16.4h, #16
                fmla v0.4s, v28.4s, v16.s[0]
                ld1 {v17.4s}, [x1], x4
                shll v17.4s, v17.4h, #16
                fmla v0.4s, v29.4s, v16.s[1]
                fmla v0.4s, v30.4s, v16.s[2]
                ld1 {v18.4s}, [x1], x4
                shll v18.4s, v18.4h, #16
                fmla v1.4s, v28.4s, v17.s[0]
                ld1 {v19.4s}, [x1], x4
                shll v19.4s, v19.4h, #16
                fmla v1.4s, v29.4s, v17.s[1]
                fmla v2.4s, v28.4s, v18.s[0]
                ld1 {v20.4s}, [x1], x4
                shll v20.4s, v20.4h, #16
                fmla v3.4s, v28.4s, v19.s[0]
                ld1 {v21.4s}, [x1], x4
                shll v21.4s, v21.4h, #16
                fmla v4.4s, v28.4s, v20.s[0]
                ld1 {v22.4s}, [x1], x4
                shll v22.4s, v22.4h, #16
                fmla v5.4s, v28.4s, v21.s[0]
                ld1 {v23.4s}, [x1], x4
                shll v23.4s, v23.4h, #16
                fmla v6.4s, v28.4s, v22.s[0]
                fmla v7.4s, v28.4s, v23.s[0]

                fmla v2.4s, v29.4s, v18.s[1]
                fmla v3.4s, v29.4s, v19.s[1]
                fmla v4.4s, v29.4s, v20.s[1]
                fmla v5.4s, v29.4s, v21.s[1]
                fmla v6.4s, v29.4s, v22.s[1]
                fmla v7.4s, v29.4s, v23.s[1]

                fmla v1.4s, v30.4s, v17.s[2]
                fmla v2.4s, v30.4s, v18.s[2]
                fmla v3.4s, v30.4s, v19.s[2]
                fmla v4.4s, v30.4s, v20.s[2]
                fmla v5.4s, v30.4s, v21.s[2]
                fmla v6.4s, v30.4s, v22.s[2]
                fmla v7.4s, v30.4s, v23.s[2]

                fmla v0.4s, v31.4s, v16.s[3]
                fmla v1.4s, v31.4s, v17.s[3]
                fmla v2.4s, v31.4s, v18.s[3]
                fmla v3.4s, v31.4s, v19.s[3]
                fmla v4.4s, v31.4s, v20.s[3]
                fmla v5.4s, v31.4s, v21.s[3]
                fmla v6.4s, v31.4s, v22.s[3]
                fmla v7.4s, v31.4s, v23.s[3]

                sub x1, x1, x14
                subs x7, x7, #1
                add x1, x1, x9
                bne L8LoopFX
            subs x8, x8, #1
            mov x7, v27.d[1]
            add x1, x1, x10
            bne L8LoopFY
        subs x5, x5, #1
        mov x8, v27.d[0]
        add x1, x1, x6
        bne L8LoopZ
    shrn v0.4h, v0.4s, #16
    shrn v1.4h, v1.4s, #16
    shrn v2.4h, v2.4s, #16
    shrn v3.4h, v3.4s, #16
    shrn v4.4h, v4.4s, #16
    shrn v5.4h, v5.4s, #16
    shrn v6.4h, v6.4s, #16
    shrn v7.4h, v7.4s, #16
    st1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x0], #32
    add x1, x11, x14
    mov x2, x12
    mov x5, x13
    sub x3, x3, #8
    cmp x3, #8
    st1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x0], #32
    bge L8Loop



L4:
cmp x3, #3
ble L1

mov x14, #4
mul x14, x4, x14

mov x11, x1
mov x12, x2
mov x13, x5
movi v0.4s, #0
movi v1.4s, #0
movi v2.4s, #0
movi v3.4s, #0
L4LoopZ:
mov v27.d[0], x8
L4LoopFY:
mov v27.d[1], x7
L4LoopFX:
ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
ld1 {v16.4s}, [x1], x4
shll v16.4s, v16.4h, #16
fmla v0.4s, v28.4s, v16.s[0]
ld1 {v17.4s}, [x1], x4
shll v17.4s, v17.4h, #16
fmla v0.4s, v29.4s, v16.s[1]
fmla v1.4s, v29.4s, v17.s[1]
ld1 {v18.4s}, [x1], x4
shll v18.4s, v18.4h, #16
fmla v1.4s, v28.4s, v17.s[0]
fmla v0.4s, v30.4s, v16.s[2]
ld1 {v19.4s}, [x1], x4
shll v19.4s, v19.4h, #16

fmla v2.4s, v28.4s, v18.s[0]
fmla v3.4s, v28.4s, v19.s[0]

fmla v2.4s, v29.4s, v18.s[1]
fmla v3.4s, v29.4s, v19.s[1]

fmla v1.4s, v30.4s, v17.s[2]
fmla v2.4s, v30.4s, v18.s[2]
fmla v3.4s, v30.4s, v19.s[2]

fmla v0.4s, v31.4s, v16.s[3]
fmla v1.4s, v31.4s, v17.s[3]
fmla v2.4s, v31.4s, v18.s[3]
fmla v3.4s, v31.4s, v19.s[3]

sub x1, x1, x14
subs x7, x7, #1
add x1, x1, x9
bne L4LoopFX
subs x8, x8, #1
mov x7, v27.d[1]
add x1, x1, x10
bne L4LoopFY
subs x5, x5, #1
mov x8, v27.d[0]
add x1, x1, x6
bne L4LoopZ
shrn v0.4h, v0.4s, #16
shrn v1.4h, v1.4s, #16
shrn v2.4h, v2.4s, #16
shrn v3.4h, v3.4s, #16
st1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x0], #32
add x1, x11, x14
mov x2, x12
mov x5, x13
sub x3, x3, #4

L1:
cmp x3, #0
ble End

L1Loop:
    mov x11, x1
    mov x12, x2
    mov x13, x5
    movi v0.4s, #0
    movi v1.4s, #0
    L1LoopZ:
        mov x14, x8
        L1LoopFY:
            mov x15, x7
            L1LoopFX:
                ld1 {v3.4s}, [x1], x9
                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                shll v3.4s, v3.4h, #16
                fmla v0.4s, v28.4s, v3.s[0]
                fmla v1.4s, v29.4s, v3.s[1]
                fmla v0.4s, v30.4s, v3.s[2]
                fmla v1.4s, v31.4s, v3.s[3]
                subs x7, x7, #1
                bne L1LoopFX
            subs x8, x8, #1
            mov x7, x15
            add x1, x1, x10
            bne L1LoopFY
        subs x5, x5, #1
        mov x8, x14
        add x1, x1, x6
        bne L1LoopZ

    fadd v0.4s, v0.4s, v1.4s
    add x1, x11, x4
    mov x2, x12
    shrn v0.4h, v0.4s, #16
    mov x5, x13
    subs x3, x3, #1
    st1 {v0.4h}, [x0], #8
    bne L1Loop

End:

sub sp, sp, #144
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ldp x19, x20, [sp], #16

ret

#endif
