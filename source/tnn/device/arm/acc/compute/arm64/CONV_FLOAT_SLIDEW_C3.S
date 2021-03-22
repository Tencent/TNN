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

asm_function GemmFloatSlidewC3
//void GemmFloatSlidewC3(float* dst,            //x0: dst 
//                          const float* src,      //x1: src
//                          const float* weight,   //x2: weight
//                          int width,             //x3: width
//                          int src_w_setup,       //x4: src_w_step
//                          int fw,                //x5: fw
//                          int fh,                //x6: fh
//                          int dilateX_step,      //x7: dilateX_step
//                          int dilateY_step);     //x8: dilateY_step, load from stack

dst          .req x0
src          .req x1
weight       .req x2
width        .req x3
src_w_step   .req x4
fw           .req x5
fh           .req x6
dilateX_step .req x7
dilateY_step .req x8
dilate_y_gap .req x10
// eor x8, x8, x8
ldr x8, [sp, #0]

//step multi by sizeof(float)
mov x12, #4
mul dilateY_step, x12, dilateY_step
mul dilateX_step, x12, dilateX_step
mul src_w_step, x12, src_w_step

//dilate_y_step -> dilate_y_step-fw*dilate_x_step
mul x12, fw, dilateX_step
sub dilate_y_gap, dilateY_step, x12

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

L14:
cmp width, #13
ble L8

mov x14, #14
mul x14, src_w_step, x14

L14Loop:
    mov x11, src
    mov x12, weight
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
    mov x9, fh
    L14LoopFY:
        mov x13, fw
        L14LoopFX:
            ld1 {v28.4s, v29.4s, v30.4s}, [weight], #48

            ld1 {v0.4s}, [src], src_w_step
            fmla v14.4s, v28.4s, v0.s[0]
            ld1 {v1.4s}, [src], src_w_step
            fmla v14.4s, v29.4s, v0.s[1]
            fmla v14.4s, v30.4s, v0.s[2]
            ld1 {v2.4s}, [src], src_w_step
            fmla v15.4s, v28.4s, v1.s[0]
            ld1 {v3.4s}, [src], src_w_step
            fmla v15.4s, v29.4s, v1.s[1]
            fmla v16.4s, v28.4s, v2.s[0]
            ld1 {v4.4s}, [src], src_w_step
            fmla v17.4s, v28.4s, v3.s[0]
            ld1 {v5.4s}, [src], src_w_step
            fmla v17.4s, v29.4s, v3.s[1]
            fmla v18.4s, v28.4s, v4.s[0]
            ld1 {v6.4s}, [src], src_w_step
            fmla v19.4s, v28.4s, v5.s[0]
            ld1 {v7.4s}, [src], src_w_step
            fmla v16.4s, v29.4s, v2.s[1]
            ld1 {v8.4s}, [src], src_w_step
            fmla v20.4s, v28.4s, v6.s[0]
            ld1 {v9.4s}, [src], src_w_step
            fmla v18.4s, v29.4s, v4.s[1]
            ld1 {v10.4s}, [src], src_w_step
            fmla v21.4s, v28.4s, v7.s[0]
            ld1 {v11.4s}, [src], src_w_step
            fmla v22.4s, v28.4s, v8.s[0]
            ld1 {v12.4s}, [src], src_w_step
            fmla v23.4s, v28.4s, v9.s[0]
            ld1 {v13.4s}, [src], src_w_step
            fmla v24.4s, v28.4s, v10.s[0]
            fmla v25.4s, v28.4s, v11.s[0]
            fmla v26.4s, v28.4s, v12.s[0]
            fmla v27.4s, v28.4s, v13.s[0]

            fmla v19.4s, v29.4s, v5.s[1]
            fmla v20.4s, v29.4s, v6.s[1]
            fmla v21.4s, v29.4s, v7.s[1]
            fmla v22.4s, v29.4s, v8.s[1]
            fmla v23.4s, v29.4s, v9.s[1]
            fmla v24.4s, v29.4s, v10.s[1]
            fmla v25.4s, v29.4s, v11.s[1]
            fmla v26.4s, v29.4s, v12.s[1]
            fmla v27.4s, v29.4s, v13.s[1]
            
            fmla v15.4s, v30.4s, v1.s[2]
            fmla v16.4s, v30.4s, v2.s[2]
            fmla v17.4s, v30.4s, v3.s[2]
            fmla v18.4s, v30.4s, v4.s[2]
            fmla v19.4s, v30.4s, v5.s[2]
            fmla v20.4s, v30.4s, v6.s[2]
            fmla v21.4s, v30.4s, v7.s[2]
            fmla v22.4s, v30.4s, v8.s[2]
            fmla v23.4s, v30.4s, v9.s[2]
            fmla v24.4s, v30.4s, v10.s[2]
            fmla v25.4s, v30.4s, v11.s[2]
            fmla v26.4s, v30.4s, v12.s[2]
            fmla v27.4s, v30.4s, v13.s[2]
            
            subs fw, fw, #1
            sub src, src, x14

            add src, src, dilateX_step
            bne L14LoopFX
        subs fh, fh, #1
        mov fw, x13
        add src, src, dilate_y_gap
        bne L14LoopFY

    mov fh, x9 
    sub width, width, #14
    st1 {v14.4s, v15.4s, v16.4s, v17.4s}, [dst], #64
    add src, x11, x14
    st1 {v18.4s, v19.4s, v20.4s, v21.4s}, [dst], #64
    mov weight, x12
    cmp width, #14
    st1 {v22.4s, v23.4s, v24.4s, v25.4s}, [dst], #64
    st1 {v26.4s, v27.4s}, [dst], #32
    bge L14Loop

L8:
cmp width, #7
ble L1

mov x14, #8
mul x14, src_w_step, x14

L8Loop:
    mov x11, src
    mov x12, weight
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0
    mov v27.d[0], fh
    L8LoopFY:
        mov v27.d[1], fw
        L8LoopFX:
            ld1 {v28.4s, v29.4s, v30.4s}, [weight], #48
            ld1 {v16.4s}, [src], src_w_step
            fmla v0.4s, v28.4s, v16.s[0]
            ld1 {v17.4s}, [src], src_w_step
            fmla v0.4s, v29.4s, v16.s[1]
            fmla v0.4s, v30.4s, v16.s[2]
            ld1 {v18.4s}, [src], src_w_step
            fmla v1.4s, v28.4s, v17.s[0]
            ld1 {v19.4s}, [src], src_w_step
            fmla v1.4s, v29.4s, v17.s[1]
            fmla v2.4s, v28.4s, v18.s[0]
            ld1 {v20.4s}, [src], src_w_step
            fmla v3.4s, v28.4s, v19.s[0]
            ld1 {v21.4s}, [src], src_w_step
            fmla v4.4s, v28.4s, v20.s[0]
            ld1 {v22.4s}, [src], src_w_step
            fmla v5.4s, v28.4s, v21.s[0]
            ld1 {v23.4s}, [src], src_w_step
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

            sub src, src, x14
            subs fw, fw, #1
            add src, src, dilateX_step
            bne L8LoopFX
        subs fh, fh, #1
        mov fw, v27.d[1]
        add src, src, dilate_y_gap
        bne L8LoopFY
    mov fh, v27.d[0]
    sub width, width, #8
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [dst], #64
    add src, x11, x14
    mov weight, x12
    cmp width, #8
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [dst], #64
    bge L8Loop

L1:
cmp width, #0
ble End

L1Loop:
    mov x11, src
    mov x12, weight
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    
    mov x14, fh
    L1LoopFY:
        mov x15, fw
        L1LoopFX:
            ld1 {v3.4s}, [src], dilateX_step
            ld1 {v28.4s, v29.4s, v30.4s}, [weight], #48

            fmla v0.4s, v28.4s, v3.s[0]
            fmla v1.4s, v29.4s, v3.s[1]
            fmla v2.4s, v30.4s, v3.s[2]
            subs fw, fw, #1
            bne L1LoopFX
        subs fh, fh, #1
        mov fw, x15
        add src, src, dilate_y_gap
        bne L1LoopFY
    mov fh, x14

    fadd v0.4s, v0.4s, v1.4s
    fadd v0.4s, v0.4s, v2.4s
    add src, x11, src_w_step
    mov weight, x12
    subs width, width, #1
    st1 {v0.4s}, [dst], #16
    bne L1Loop

End:

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

ret

#endif
