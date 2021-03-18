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

asm_function DeconvFp16O8
//void DeconvFp16O8(__fp16* dst,             // x0
//                     const __fp16* src,    // x1
//                     const __fp16* weight, // x2
//                     int width,            // x3
//                     int dst_w_step,       // x4
//                     int src_depth_div8,   // x5
//                     int src_depth_step,   // x6
//                     int fw,               // x7
//                     int fh,               // x8
//                     int dilate_x_step,    // x9
//                     int dilate_y_step)    // x10

dst          .req x0
src          .req x1
weight       .req x2
width        .req x3
dst_w_step   .req x4
ic8          .req x5
fw           .req x7
fh           .req x8
dilate_x_step .req x9
dilate_y_step .req x10
dst_tmp      .req x15

//Auto Load:
//x0:dst, x1:src, x2:weight, x3:width, x4:src_w_step, x5:src_depth_div8, x6: src_depth_step, x7:fw

//Load from sp
//x8:fh, x9:dilate_x_step, x10:dilate_y_step
// eor x8, x8, x8
ldr x8, [sp, #0]
// eor x9, x9, x9
ldr x9, [sp, #8]
// eor x10, x10, x10
ldr x10, [sp, #16]

//step multi by sizeof(__fp16)
lsl x10, x10, #1
lsl x9, x9, #1
lsl x6, x6, #1
lsl x4, x4, #1

//src_depth_step -> src_depth_step - fh*dilate_y_step
//mul x12, x8, x10
//sub x6, x6, x12

//dilate_y_step -> dilate_y_step-fw*dilate_x_step
//mul x12, x7, x9
//sub x10, x10, x12

sub sp, sp, #144
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
stp x19, x20, [sp], #16

L14:
cmp x3, #13
ble L4

L14Loop:
    mov x11, src 
    mov x12, weight 

    mov x14, #14
    mul x14, dst_w_step, x14
    mov x19, fh
    L14LoopFY:
        mov x20, fw
        L14LoopFX:
            mov x13, ic8
            ld1 {v28.8h, v29.8h, v30.8h, v31.8h}, [weight], #64
            ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x1], #64
            ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x1], #64
            ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x1], #64
            ld1 {v12.8h, v13.8h}, [x1], #32

            fmul v14.8h, v28.8h, v0.h[0]
            fmul v15.8h, v28.8h, v1.h[0]
            fmul v16.8h, v28.8h, v2.h[0]
            fmul v17.8h, v28.8h, v3.h[0]
            fmul v18.8h, v28.8h, v4.h[0]
            fmul v19.8h, v28.8h, v5.h[0]
            fmul v20.8h, v28.8h, v6.h[0]
            fmul v21.8h, v28.8h, v7.h[0]
            fmul v22.8h, v28.8h, v8.h[0]
            fmul v23.8h, v28.8h, v9.h[0]
            fmul v24.8h, v28.8h, v10.h[0]
            fmul v25.8h, v28.8h, v11.h[0]
            fmul v26.8h, v28.8h, v12.h[0]
            fmul v27.8h, v28.8h, v13.h[0]

            subs ic8, ic8, #1
            beq L14LoopZEnd
            L14LoopZ:
                ld1 {v28.8h}, [weight], #16
                sub src, src, #224
                add src, src, x6

                fmla v14.8h, v29.8h, v0.h[1]
                fmla v15.8h, v29.8h, v1.h[1]
                fmla v16.8h, v29.8h, v2.h[1]
                fmla v17.8h, v29.8h, v3.h[1]
                fmla v18.8h, v29.8h, v4.h[1]
                fmla v19.8h, v29.8h, v5.h[1]
                fmla v20.8h, v29.8h, v6.h[1]
                fmla v21.8h, v29.8h, v7.h[1]
                fmla v22.8h, v29.8h, v8.h[1]
                fmla v23.8h, v29.8h, v9.h[1]
                fmla v24.8h, v29.8h, v10.h[1]
                fmla v25.8h, v29.8h, v11.h[1]
                fmla v26.8h, v29.8h, v12.h[1]
                fmla v27.8h, v29.8h, v13.h[1]

                ld1 {v29.8h}, [weight], #16
                fmla v14.8h, v30.8h, v0.h[2]
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

                ld1 {v30.8h}, [weight], #16
                fmla v14.8h, v31.8h, v0.h[3]
                fmla v15.8h, v31.8h, v1.h[3]
                fmla v16.8h, v31.8h, v2.h[3]
                fmla v17.8h, v31.8h, v3.h[3]
                fmla v18.8h, v31.8h, v4.h[3]
                fmla v19.8h, v31.8h, v5.h[3]
                fmla v20.8h, v31.8h, v6.h[3]
                fmla v21.8h, v31.8h, v7.h[3]
                fmla v22.8h, v31.8h, v8.h[3]
                fmla v23.8h, v31.8h, v9.h[3]
                fmla v24.8h, v31.8h, v10.h[3]
                fmla v25.8h, v31.8h, v11.h[3]
                fmla v26.8h, v31.8h, v12.h[3]
                fmla v27.8h, v31.8h, v13.h[3]

                ld1 {v31.8h}, [weight], #16
                fmla v14.8h, v28.8h, v0.h[4]
                fmla v15.8h, v28.8h, v1.h[4]
                fmla v16.8h, v28.8h, v2.h[4]
                fmla v17.8h, v28.8h, v3.h[4]
                fmla v18.8h, v28.8h, v4.h[4]
                fmla v19.8h, v28.8h, v5.h[4]
                fmla v20.8h, v28.8h, v6.h[4]
                fmla v21.8h, v28.8h, v7.h[4]
                fmla v22.8h, v28.8h, v8.h[4]
                fmla v23.8h, v28.8h, v9.h[4]
                fmla v24.8h, v28.8h, v10.h[4]
                fmla v25.8h, v28.8h, v11.h[4]
                fmla v26.8h, v28.8h, v12.h[4]
                fmla v27.8h, v28.8h, v13.h[4]

                ld1 {v28.8h}, [weight], #16
                fmla v14.8h, v29.8h, v0.h[5]
                fmla v15.8h, v29.8h, v1.h[5]
                fmla v16.8h, v29.8h, v2.h[5]
                fmla v17.8h, v29.8h, v3.h[5]
                fmla v18.8h, v29.8h, v4.h[5]
                fmla v19.8h, v29.8h, v5.h[5]
                fmla v20.8h, v29.8h, v6.h[5]
                fmla v21.8h, v29.8h, v7.h[5]
                fmla v22.8h, v29.8h, v8.h[5]
                fmla v23.8h, v29.8h, v9.h[5]
                fmla v24.8h, v29.8h, v10.h[5]
                fmla v25.8h, v29.8h, v11.h[5]
                fmla v26.8h, v29.8h, v12.h[5]
                fmla v27.8h, v29.8h, v13.h[5]

                ld1 {v29.8h}, [weight], #16
                fmla v14.8h, v30.8h, v0.h[6]
                fmla v15.8h, v30.8h, v1.h[6]
                fmla v16.8h, v30.8h, v2.h[6]
                fmla v17.8h, v30.8h, v3.h[6]
                fmla v18.8h, v30.8h, v4.h[6]
                fmla v19.8h, v30.8h, v5.h[6]
                fmla v20.8h, v30.8h, v6.h[6]
                fmla v21.8h, v30.8h, v7.h[6]
                fmla v22.8h, v30.8h, v8.h[6]
                fmla v23.8h, v30.8h, v9.h[6]
                fmla v24.8h, v30.8h, v10.h[6]
                fmla v25.8h, v30.8h, v11.h[6]
                fmla v26.8h, v30.8h, v12.h[6]
                fmla v27.8h, v30.8h, v13.h[6]

                ld1 {v30.8h}, [weight], #16
                fmla v14.8h, v31.8h, v0.h[7]
                fmla v15.8h, v31.8h, v1.h[7]
                fmla v16.8h, v31.8h, v2.h[7]
                fmla v17.8h, v31.8h, v3.h[7]
                fmla v18.8h, v31.8h, v4.h[7]
                fmla v19.8h, v31.8h, v5.h[7]
                fmla v20.8h, v31.8h, v6.h[7]
                fmla v21.8h, v31.8h, v7.h[7]
                fmla v22.8h, v31.8h, v8.h[7]
                fmla v23.8h, v31.8h, v9.h[7]
                fmla v24.8h, v31.8h, v10.h[7]
                fmla v25.8h, v31.8h, v11.h[7]
                fmla v26.8h, v31.8h, v12.h[7]
                fmla v27.8h, v31.8h, v13.h[7]

                ld1 {v31.8h}, [weight], #16
                ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x1], #64
                ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x1], #64
                ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x1], #64
                ld1 {v12.8h, v13.8h}, [x1], #32

                subs ic8, ic8, #1
                fmla v14.8h, v28.8h, v0.h[0]
                fmla v15.8h, v28.8h, v1.h[0]
                fmla v16.8h, v28.8h, v2.h[0]
                fmla v17.8h, v28.8h, v3.h[0]
                fmla v18.8h, v28.8h, v4.h[0]
                fmla v19.8h, v28.8h, v5.h[0]
                fmla v20.8h, v28.8h, v6.h[0]
                fmla v21.8h, v28.8h, v7.h[0]
                fmla v22.8h, v28.8h, v8.h[0]
                fmla v23.8h, v28.8h, v9.h[0]
                fmla v24.8h, v28.8h, v10.h[0]
                fmla v25.8h, v28.8h, v11.h[0]
                fmla v26.8h, v28.8h, v12.h[0]
                fmla v27.8h, v28.8h, v13.h[0]

                bne L14LoopZ

            L14LoopZEnd:
            ld1 {v28.8h}, [weight], #16
            fmla v14.8h, v29.8h, v0.h[1]
            fmla v15.8h, v29.8h, v1.h[1]
            fmla v16.8h, v29.8h, v2.h[1]
            fmla v17.8h, v29.8h, v3.h[1]
            fmla v18.8h, v29.8h, v4.h[1]
            fmla v19.8h, v29.8h, v5.h[1]
            fmla v20.8h, v29.8h, v6.h[1]
            fmla v21.8h, v29.8h, v7.h[1]
            fmla v22.8h, v29.8h, v8.h[1]
            fmla v23.8h, v29.8h, v9.h[1]
            fmla v24.8h, v29.8h, v10.h[1]
            fmla v25.8h, v29.8h, v11.h[1]
            fmla v26.8h, v29.8h, v12.h[1]
            fmla v27.8h, v29.8h, v13.h[1]

            ld1 {v29.8h}, [weight], #16
            fmla v14.8h, v30.8h, v0.h[2]
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

            ld1 {v30.8h}, [weight], #16
            fmla v14.8h, v31.8h, v0.h[3]
            fmla v15.8h, v31.8h, v1.h[3]
            fmla v16.8h, v31.8h, v2.h[3]
            fmla v17.8h, v31.8h, v3.h[3]
            fmla v18.8h, v31.8h, v4.h[3]
            fmla v19.8h, v31.8h, v5.h[3]
            fmla v20.8h, v31.8h, v6.h[3]
            fmla v21.8h, v31.8h, v7.h[3]
            fmla v22.8h, v31.8h, v8.h[3]
            fmla v23.8h, v31.8h, v9.h[3]
            fmla v24.8h, v31.8h, v10.h[3]
            fmla v25.8h, v31.8h, v11.h[3]
            fmla v26.8h, v31.8h, v12.h[3]
            fmla v27.8h, v31.8h, v13.h[3]

            ld1 {v31.8h}, [weight], #16
            fmla v14.8h, v28.8h, v0.h[4]
            fmla v15.8h, v28.8h, v1.h[4]
            fmla v16.8h, v28.8h, v2.h[4]
            fmla v17.8h, v28.8h, v3.h[4]
            fmla v18.8h, v28.8h, v4.h[4]
            fmla v19.8h, v28.8h, v5.h[4]
            fmla v20.8h, v28.8h, v6.h[4]
            fmla v21.8h, v28.8h, v7.h[4]
            fmla v22.8h, v28.8h, v8.h[4]
            fmla v23.8h, v28.8h, v9.h[4]
            fmla v24.8h, v28.8h, v10.h[4]
            fmla v25.8h, v28.8h, v11.h[4]
            fmla v26.8h, v28.8h, v12.h[4]
            fmla v27.8h, v28.8h, v13.h[4]

            fmla v14.8h, v29.8h, v0.h[5]
            fmla v15.8h, v29.8h, v1.h[5]
            fmla v16.8h, v29.8h, v2.h[5]
            fmla v17.8h, v29.8h, v3.h[5]
            fmla v18.8h, v29.8h, v4.h[5]
            fmla v19.8h, v29.8h, v5.h[5]
            fmla v20.8h, v29.8h, v6.h[5]
            fmla v21.8h, v29.8h, v7.h[5]
            fmla v22.8h, v29.8h, v8.h[5]
            fmla v23.8h, v29.8h, v9.h[5]
            fmla v24.8h, v29.8h, v10.h[5]
            fmla v25.8h, v29.8h, v11.h[5]
            fmla v26.8h, v29.8h, v12.h[5]
            fmla v27.8h, v29.8h, v13.h[5]

            mov dst_tmp, dst
            fmla v14.8h, v30.8h, v0.h[6]
            fmla v15.8h, v30.8h, v1.h[6]
            fmla v16.8h, v30.8h, v2.h[6]
            fmla v17.8h, v30.8h, v3.h[6]
            fmla v18.8h, v30.8h, v4.h[6]
            fmla v19.8h, v30.8h, v5.h[6]
            fmla v20.8h, v30.8h, v6.h[6]
            fmla v21.8h, v30.8h, v7.h[6]
            fmla v22.8h, v30.8h, v8.h[6]
            fmla v23.8h, v30.8h, v9.h[6]
            fmla v24.8h, v30.8h, v10.h[6]
            fmla v25.8h, v30.8h, v11.h[6]
            fmla v26.8h, v30.8h, v12.h[6]
            fmla v27.8h, v30.8h, v13.h[6]

            fmla v14.8h, v31.8h, v0.h[7]
            ld1 {v0.8h}, [dst_tmp], dst_w_step
            fmla v15.8h, v31.8h, v1.h[7]
            ld1 {v1.8h}, [dst_tmp], dst_w_step
            fmla v16.8h, v31.8h, v2.h[7]
            ld1 {v2.8h}, [dst_tmp], dst_w_step
            fmla v17.8h, v31.8h, v3.h[7]
            ld1 {v3.8h}, [dst_tmp], dst_w_step
            fmla v18.8h, v31.8h, v4.h[7]
            ld1 {v4.8h}, [dst_tmp], dst_w_step
            fmla v19.8h, v31.8h, v5.h[7]
            ld1 {v5.8h}, [dst_tmp], dst_w_step
            fmla v20.8h, v31.8h, v6.h[7]
            ld1 {v6.8h}, [dst_tmp], dst_w_step
            fmla v21.8h, v31.8h, v7.h[7]
            ld1 {v7.8h}, [dst_tmp], dst_w_step
            fmla v22.8h, v31.8h, v8.h[7]
            ld1 {v8.8h}, [dst_tmp], dst_w_step
            fmla v23.8h, v31.8h, v9.h[7]
            ld1 {v9.8h}, [dst_tmp], dst_w_step
            fmla v24.8h, v31.8h, v10.h[7]
            ld1 {v10.8h}, [dst_tmp], dst_w_step
            fmla v25.8h, v31.8h, v11.h[7]
            ld1 {v11.8h}, [dst_tmp], dst_w_step
            fmla v26.8h, v31.8h, v12.h[7]
            ld1 {v12.8h}, [dst_tmp], dst_w_step
            fmla v27.8h, v31.8h, v13.h[7]
            ld1 {v13.8h}, [dst_tmp], dst_w_step

            // add with stride
            fadd v14.8h, v14.8h, v0.8h
            fadd v15.8h, v15.8h, v1.8h
            fadd v16.8h, v16.8h, v2.8h
            fadd v17.8h, v17.8h, v3.8h
            fadd v18.8h, v18.8h, v4.8h
            fadd v19.8h, v19.8h, v5.8h
            fadd v20.8h, v20.8h, v6.8h
            fadd v21.8h, v21.8h, v7.8h
            fadd v22.8h, v22.8h, v8.8h
            fadd v23.8h, v23.8h, v9.8h
            fadd v24.8h, v24.8h, v10.8h
            fadd v25.8h, v25.8h, v11.8h
            fadd v26.8h, v26.8h, v12.8h
            fadd v27.8h, v27.8h, v13.8h

            st1 {v14.8h}, [dst], dst_w_step
            st1 {v15.8h}, [dst], dst_w_step
            st1 {v16.8h}, [dst], dst_w_step
            st1 {v17.8h}, [dst], dst_w_step
            st1 {v18.8h}, [dst], dst_w_step
            st1 {v19.8h}, [dst], dst_w_step
            st1 {v20.8h}, [dst], dst_w_step
            st1 {v21.8h}, [dst], dst_w_step
            st1 {v22.8h}, [dst], dst_w_step
            st1 {v23.8h}, [dst], dst_w_step
            st1 {v24.8h}, [dst], dst_w_step
            st1 {v25.8h}, [dst], dst_w_step
            st1 {v26.8h}, [dst], dst_w_step
            st1 {v27.8h}, [dst], dst_w_step

            sub dst, dst, x14
            add dst, dst, dilate_x_step

            mov ic8, x13
            subs fw, fw, #1
            sub x1, x1, x14
            mov src, x11
            bne L14LoopFX
        subs fh, fh, #1
        mov fw, x20
        mul x20, fw, dilate_x_step
        sub dst, dst, x20
        add dst, dst, dilate_y_step
        bne L14LoopFY

    mov fh, x19
    mul x20, fh, dilate_y_step
    sub dst, dst, x20
    add src, src, #224
    add dst, dst, x14
    mov weight, x12 
    sub width, width, #14
    cmp width, #14
    bge L14Loop

L4:
cmp x3, #3
ble L1

L4Loop:
    mov x11, src 
    mov x12, weight 

    mov x14, #4
    mul x14, x14, dst_w_step

    mov x19, fh
    L4LoopFY:
        mov x20, fw
        L4LoopFX:
            mov x13, ic8
            ld1 {v28.8h, v29.8h, v30.8h, v31.8h}, [weight], #64
            ldr q0, [src]
            ldr q1, [src, #16]
            ldr q2, [src, #32]
            ldr q3, [src, #48]
            fmul v14.8h, v28.8h, v0.h[0]
            fmul v15.8h, v28.8h, v1.h[0]
            fmul v16.8h, v28.8h, v2.h[0]
            fmul v17.8h, v28.8h, v3.h[0]

            subs ic8, ic8, #1
            beq L4LoopZEnd
            L4LoopZ:
                add src, src, x6

                ld1 {v28.8h}, [weight], #16
                fmla v14.8h, v29.8h, v0.h[1]
                fmla v15.8h, v29.8h, v1.h[1]
                fmla v16.8h, v29.8h, v2.h[1]
                fmla v17.8h, v29.8h, v3.h[1]

                ld1 {v29.8h}, [weight], #16
                fmla v14.8h, v30.8h, v0.h[2]
                fmla v15.8h, v30.8h, v1.h[2]
                fmla v16.8h, v30.8h, v2.h[2]
                fmla v17.8h, v30.8h, v3.h[2]

                ld1 {v30.8h}, [weight], #16
                fmla v14.8h, v31.8h, v0.h[3]
                fmla v15.8h, v31.8h, v1.h[3]
                fmla v16.8h, v31.8h, v2.h[3]
                fmla v17.8h, v31.8h, v3.h[3]

                ld1 {v31.8h}, [weight], #16
                fmla v14.8h, v28.8h, v0.h[4]
                fmla v15.8h, v28.8h, v1.h[4]
                fmla v16.8h, v28.8h, v2.h[4]
                fmla v17.8h, v28.8h, v3.h[4]

                ld1 {v28.8h}, [weight], #16
                fmla v14.8h, v29.8h, v0.h[5]
                fmla v15.8h, v29.8h, v1.h[5]
                fmla v16.8h, v29.8h, v2.h[5]
                fmla v17.8h, v29.8h, v3.h[5]

                ld1 {v29.8h}, [weight], #16
                fmla v14.8h, v30.8h, v0.h[6]
                fmla v15.8h, v30.8h, v1.h[6]
                fmla v16.8h, v30.8h, v2.h[6]
                fmla v17.8h, v30.8h, v3.h[6]

                ld1 {v30.8h}, [weight], #16
                fmla v14.8h, v31.8h, v0.h[7]
                fmla v15.8h, v31.8h, v1.h[7]
                fmla v16.8h, v31.8h, v2.h[7]
                fmla v17.8h, v31.8h, v3.h[7]

                ldr q0, [src]
                ldr q1, [src, #16]
                ldr q2, [src, #32]
                ldr q3, [src, #48]

                ld1 {v31.8h}, [weight], #16
                fmla v14.8h, v28.8h, v0.h[0]
                fmla v15.8h, v28.8h, v1.h[0]
                fmla v16.8h, v28.8h, v2.h[0]
                fmla v17.8h, v28.8h, v3.h[0]

                subs ic8, ic8, #1
                bne L4LoopZ

            L4LoopZEnd:
            ld1 {v28.8h}, [weight], #16
            fmla v14.8h, v29.8h, v0.h[1]
            fmla v15.8h, v29.8h, v1.h[1]
            fmla v16.8h, v29.8h, v2.h[1]
            fmla v17.8h, v29.8h, v3.h[1]

            ld1 {v29.8h}, [weight], #16
            fmla v14.8h, v30.8h, v0.h[2]
            fmla v15.8h, v30.8h, v1.h[2]
            fmla v16.8h, v30.8h, v2.h[2]
            fmla v17.8h, v30.8h, v3.h[2]

            ld1 {v30.8h}, [weight], #16
            fmla v14.8h, v31.8h, v0.h[3]
            fmla v15.8h, v31.8h, v1.h[3]
            fmla v16.8h, v31.8h, v2.h[3]
            fmla v17.8h, v31.8h, v3.h[3]

            ld1 {v31.8h}, [weight], #16
            fmla v14.8h, v28.8h, v0.h[4]
            fmla v15.8h, v28.8h, v1.h[4]
            fmla v16.8h, v28.8h, v2.h[4]
            fmla v17.8h, v28.8h, v3.h[4]

            mov dst_tmp, dst
            fmla v14.8h, v29.8h, v0.h[5]
            fmla v15.8h, v29.8h, v1.h[5]
            fmla v16.8h, v29.8h, v2.h[5]
            fmla v17.8h, v29.8h, v3.h[5]

            fmla v14.8h, v30.8h, v0.h[6]
            fmla v15.8h, v30.8h, v1.h[6]
            fmla v16.8h, v30.8h, v2.h[6]
            fmla v17.8h, v30.8h, v3.h[6]

            fmla v14.8h, v31.8h, v0.h[7]
            ld1 {v0.8h}, [dst_tmp], dst_w_step
            fmla v15.8h, v31.8h, v1.h[7]
            ld1 {v1.8h}, [dst_tmp], dst_w_step
            fmla v16.8h, v31.8h, v2.h[7]
            ld1 {v2.8h}, [dst_tmp], dst_w_step
            fmla v17.8h, v31.8h, v3.h[7]
            ld1 {v3.8h}, [dst_tmp], dst_w_step

            // add with stride
            fadd v14.8h, v14.8h, v0.8h
            fadd v15.8h, v15.8h, v1.8h
            fadd v16.8h, v16.8h, v2.8h
            fadd v17.8h, v17.8h, v3.8h
            st1 {v14.8h}, [dst], dst_w_step
            st1 {v15.8h}, [dst], dst_w_step
            st1 {v16.8h}, [dst], dst_w_step
            st1 {v17.8h}, [dst], dst_w_step

            sub dst, dst, x14
            add dst, dst, dilate_x_step

            mov ic8, x13
            subs fw, fw, #1
            sub x1, x1, x14
            mov src, x11
            bne L4LoopFX
        subs fh, fh, #1
        mov fw, x20
        mul x20, fw, dilate_x_step
        sub dst, dst, x20
        add dst, dst, dilate_y_step
        bne L4LoopFY
    
    mov fh, x19
    mul x20, fh, dilate_y_step
    sub dst, dst, x20
    add src, src, #64
    add dst, dst, x14
    mov weight, x12 
    sub width, width, #4
    cmp width, #4
    bge L4Loop


L1:
cmp x3, #0
ble End

L1Loop:
    mov x11, src 
    mov x12, weight 

    mov x14, #1
    mul x14, dst_w_step, x14

    mov x19, fh
    L1LoopFY:
        mov x20, fw
        L1LoopFX:
            mov x13, ic8
            eor v14.16b, v14.16b, v14.16b
            eor v15.16b, v15.16b, v15.16b
            eor v16.16b, v16.16b, v16.16b
            eor v17.16b, v17.16b, v17.16b
            L1LoopZ:
                ld1 {v20.8h, v21.8h, v22.8h, v23.8h}, [weight], #64
                ld1 {v24.8h, v25.8h, v26.8h, v27.8h}, [weight], #64
                ld1 {v0.8h}, [src], x6

                fmla v14.8h, v20.8h, v0.h[0]
                fmla v15.8h, v21.8h, v0.h[1]
                fmla v16.8h, v22.8h, v0.h[2]
                fmla v17.8h, v23.8h, v0.h[3]
                fmla v14.8h, v24.8h, v0.h[4]
                fmla v15.8h, v25.8h, v0.h[5]
                fmla v16.8h, v26.8h, v0.h[6]
                fmla v17.8h, v27.8h, v0.h[7]

                subs ic8, ic8, #1
                bne L1LoopZ

            L1LoopZEnd:
            ld1 {v0.8h}, [dst]
            // add with stride
            fadd v14.8h, v14.8h, v15.8h
            fadd v16.8h, v16.8h, v17.8h
            fadd v14.8h, v14.8h, v16.8h
            fadd v14.8h, v14.8h, v0.8h
            st1 {v14.8h}, [dst], dst_w_step

            sub dst, dst, x14
            add dst, dst, dilate_x_step

            mov ic8, x13
            subs fw, fw, #1
            sub x1, x1, x14
            mov src, x11
            bne L1LoopFX
        subs fh, fh, #1
        mov fw, x20
        mul x20, fw, dilate_x_step
        sub dst, dst, x20
        add dst, dst, dilate_y_step
        bne L1LoopFY

    mov fh, x19
    mul x20, fh, dilate_y_step
    sub dst, dst, x20
    add src, src, #16
    add dst, dst, x14
    mov weight, x12 
    sub width, width, #1
    cmp width, #1
    bge L1Loop

End:

sub sp, sp, #144
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ldp x19, x20, [sp], #16

ret

#endif
#endif
