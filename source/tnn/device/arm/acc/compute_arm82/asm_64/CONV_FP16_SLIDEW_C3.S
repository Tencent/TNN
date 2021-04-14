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

asm_function GemmFp16SlidewC3
//void GemmFp16SlidewC3(__fp16* dst,            //x0: dst 
//                          const __fp16* src,      //x1: src
//                          const __fp16* weight,   //x2: weight
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

//step multi by sizeof(__fp16)
lsl dilateY_step, dilateY_step, #1
lsl dilateX_step, dilateX_step, #1
lsl src_w_step, src_w_step, #1

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
    movi v14.8h, #0
    movi v15.8h, #0
    movi v16.8h, #0
    movi v17.8h, #0
    movi v18.8h, #0
    movi v19.8h, #0
    movi v20.8h, #0
    movi v21.8h, #0
    movi v22.8h, #0
    movi v23.8h, #0
    movi v24.8h, #0
    movi v25.8h, #0
    movi v26.8h, #0
    movi v27.8h, #0
    mov x9, fh
    L14LoopFY:
        mov x13, fw
        L14LoopFX:
            ld1 {v28.8h, v29.8h, v30.8h}, [weight], #48

            ld1 {v0.8h}, [src], src_w_step
            fmla v14.8h, v28.8h, v0.h[0]
            ld1 {v1.8h}, [src], src_w_step
            fmla v14.8h, v29.8h, v0.h[1]
            fmla v14.8h, v30.8h, v0.h[2]
            ld1 {v2.8h}, [src], src_w_step
            fmla v15.8h, v28.8h, v1.h[0]
            ld1 {v3.8h}, [src], src_w_step
            fmla v15.8h, v29.8h, v1.h[1]
            fmla v16.8h, v28.8h, v2.h[0]
            ld1 {v4.8h}, [src], src_w_step
            fmla v17.8h, v28.8h, v3.h[0]
            ld1 {v5.8h}, [src], src_w_step
            fmla v17.8h, v29.8h, v3.h[1]
            fmla v18.8h, v28.8h, v4.h[0]
            ld1 {v6.8h}, [src], src_w_step
            fmla v19.8h, v28.8h, v5.h[0]
            ld1 {v7.8h}, [src], src_w_step
            fmla v16.8h, v29.8h, v2.h[1]
            ld1 {v8.8h}, [src], src_w_step
            fmla v20.8h, v28.8h, v6.h[0]
            ld1 {v9.8h}, [src], src_w_step
            fmla v18.8h, v29.8h, v4.h[1]
            ld1 {v10.8h}, [src], src_w_step
            fmla v21.8h, v28.8h, v7.h[0]
            ld1 {v11.8h}, [src], src_w_step
            fmla v22.8h, v28.8h, v8.h[0]
            ld1 {v12.8h}, [src], src_w_step
            fmla v23.8h, v28.8h, v9.h[0]
            ld1 {v13.8h}, [src], src_w_step
            fmla v24.8h, v28.8h, v10.h[0]
            fmla v25.8h, v28.8h, v11.h[0]
            fmla v26.8h, v28.8h, v12.h[0]
            fmla v27.8h, v28.8h, v13.h[0]

            fmla v19.8h, v29.8h, v5.h[1]
            fmla v20.8h, v29.8h, v6.h[1]
            fmla v21.8h, v29.8h, v7.h[1]
            fmla v22.8h, v29.8h, v8.h[1]
            fmla v23.8h, v29.8h, v9.h[1]
            fmla v24.8h, v29.8h, v10.h[1]
            fmla v25.8h, v29.8h, v11.h[1]
            fmla v26.8h, v29.8h, v12.h[1]
            fmla v27.8h, v29.8h, v13.h[1]
            
            fmla v15.8h, v30.8h, v1.h[2]
            fmla v16.8h, v30.8h, v2.h[2]
            fmla v17.8h, v30.8h, v3.h[2]
            fmla v18.8h, v30.8h, v4.h[2]
            fmla v19.8h, v30.8h, v5.h[2]
            fmla v20.8h, v30.8h, v6.h[2]
            fmla v21.8h, v30.8h, v7.h[2]
            fmla v22.8h, v30.8h, v8.h[2]
            fmla v23.8h, v30.8h, v9.h[2]
            fmla v24.8h, v30.8h, v10.h[2]
            fmla v25.8h, v30.8h, v11.h[2]
            fmla v26.8h, v30.8h, v12.h[2]
            fmla v27.8h, v30.8h, v13.h[2]
            
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
    st1 {v14.8h, v15.8h, v16.8h, v17.8h}, [dst], #64
    add src, x11, x14
    st1 {v18.8h, v19.8h, v20.8h, v21.8h}, [dst], #64
    mov weight, x12
    cmp width, #14
    st1 {v22.8h, v23.8h, v24.8h, v25.8h}, [dst], #64
    st1 {v26.8h, v27.8h}, [dst], #32
    bge L14Loop

L8:
cmp width, #7
ble L1

mov x14, #8
mul x14, src_w_step, x14

L8Loop:
    mov x11, src
    mov x12, weight
    movi v16.8h, #0
    movi v17.8h, #0
    movi v18.8h, #0
    movi v19.8h, #0
    movi v20.8h, #0
    movi v21.8h, #0
    movi v22.8h, #0
    movi v23.8h, #0
    mov v27.d[0], fh
    L8LoopFY:
        mov v27.d[1], fw
        L8LoopFX:
            ld1 {v28.8h, v29.8h, v30.8h}, [weight], #48
            ld1 {v0.8h}, [src], src_w_step
            fmla v16.8h, v28.8h, v0.h[0]
            ld1 {v1.8h}, [src], src_w_step
            fmla v16.8h, v29.8h, v0.h[1]
            fmla v16.8h, v30.8h, v0.h[2]
            ld1 {v2.8h}, [src], src_w_step
            fmla v17.8h, v28.8h, v1.h[0]
            ld1 {v3.8h}, [src], src_w_step
            fmla v17.8h, v29.8h, v1.h[1]
            fmla v18.8h, v28.8h, v2.h[0]
            ld1 {v4.8h}, [src], src_w_step
            fmla v19.8h, v28.8h, v3.h[0]
            ld1 {v5.8h}, [src], src_w_step
            fmla v20.8h, v28.8h, v4.h[0]
            ld1 {v6.8h}, [src], src_w_step
            fmla v21.8h, v28.8h, v5.h[0]
            ld1 {v7.8h}, [src], src_w_step
            fmla v22.8h, v28.8h, v6.h[0]
            fmla v23.8h, v28.8h, v7.h[0]

            fmla v18.8h, v29.8h, v2.h[1]
            fmla v19.8h, v29.8h, v3.h[1]
            fmla v20.8h, v29.8h, v4.h[1]
            fmla v21.8h, v29.8h, v5.h[1]
            fmla v22.8h, v29.8h, v6.h[1]
            fmla v23.8h, v29.8h, v7.h[1]

            fmla v17.8h, v30.8h, v1.h[2]
            fmla v18.8h, v30.8h, v2.h[2]
            fmla v19.8h, v30.8h, v3.h[2]
            fmla v20.8h, v30.8h, v4.h[2]
            fmla v21.8h, v30.8h, v5.h[2]
            fmla v22.8h, v30.8h, v6.h[2]
            fmla v23.8h, v30.8h, v7.h[2]

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
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [dst], #64
    add src, x11, x14
    mov weight, x12
    cmp width, #8
    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [dst], #64
    bge L8Loop

L1:
cmp width, #0
ble End

L1Loop:
    mov x11, src
    mov x12, weight
    movi v0.8h, #0
    movi v1.8h, #0
    movi v2.8h, #0
    
    mov x14, fh
    L1LoopFY:
        mov x15, fw
        L1LoopFX:
            ld1 {v3.8h}, [src], dilateX_step
            ld1 {v28.8h, v29.8h, v30.8h}, [weight], #48

            fmla v0.8h, v28.8h, v3.h[0]
            fmla v1.8h, v29.8h, v3.h[1]
            fmla v2.8h, v30.8h, v3.h[2]
            subs fw, fw, #1
            bne L1LoopFX
        subs fh, fh, #1
        mov fw, x15
        add src, src, dilate_y_gap
        bne L1LoopFY
    mov fh, x14

    fadd v0.8h, v0.8h, v1.8h
    fadd v0.8h, v0.8h, v2.8h
    add src, x11, src_w_step
    mov weight, x12
    subs width, width, #1
    st1 {v0.8h}, [dst], #16
    bne L1Loop

End:

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

ret

#endif
#endif
