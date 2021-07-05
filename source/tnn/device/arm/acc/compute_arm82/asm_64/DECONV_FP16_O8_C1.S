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

asm_function DeconvFp16O8C1
//input is nchw
//output is nc8hw8
//void DeconvFp16O8C1(__fp16* dst,             // x0
//                     const __fp16* src,    // x1
//                     const __fp16* weight, // x2
//                     int width,            // x3
//                     int dst_w_step,       // x4
//                     int src_depth,        // x5
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
input_c      .req x5
fw           .req x7
fh           .req x8
dilate_x_step .req x9
dilate_y_step .req x10
dst_tmp      .req x15

//Auto Load:
//x0:dst, x1:src, x2:weight, x3:width, x4:src_w_step, x5:src_depth, x6: src_depth_step, x7:fw

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

sub sp, sp, #144
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
stp x19, x20, [sp], #16

L16:
cmp x3, #15
ble L4

L16Loop:
    mov x11, src 
    mov x12, weight 

    mov x14, #16
    mul x14, dst_w_step, x14
    mov x19, fh
    L16LoopFY:
        mov x20, fw
        L16LoopFX:
            mov x13, input_c
            ld1 {v4.8h}, [weight], #16
            ld1 {v0.8h, v1.8h}, [src], x6

            fmul v16.8h, v4.8h, v0.h[0]
            fmul v17.8h, v4.8h, v0.h[1]
            fmul v18.8h, v4.8h, v0.h[2]
            fmul v19.8h, v4.8h, v0.h[3]
            fmul v20.8h, v4.8h, v0.h[4]
            fmul v21.8h, v4.8h, v0.h[5]
            fmul v22.8h, v4.8h, v0.h[6]
            fmul v23.8h, v4.8h, v0.h[7]
            fmul v24.8h, v4.8h, v1.h[0]
            fmul v25.8h, v4.8h, v1.h[1]
            fmul v26.8h, v4.8h, v1.h[2]
            fmul v27.8h, v4.8h, v1.h[3]
            fmul v28.8h, v4.8h, v1.h[4]
            fmul v29.8h, v4.8h, v1.h[5]
            fmul v30.8h, v4.8h, v1.h[6]
            fmul v31.8h, v4.8h, v1.h[7]

            subs input_c, input_c, #1
            beq L16LoopZEnd
            L16LoopZ:
                ld1 {v4.8h}, [weight], #16
                ld1 {v0.8h, v1.8h}, [src], x6

                subs input_c, input_c, #1
                fmla v16.8h, v4.8h, v0.h[0]
                fmla v17.8h, v4.8h, v0.h[1]
                fmla v18.8h, v4.8h, v0.h[2]
                fmla v19.8h, v4.8h, v0.h[3]
                fmla v20.8h, v4.8h, v0.h[4]
                fmla v21.8h, v4.8h, v0.h[5]
                fmla v22.8h, v4.8h, v0.h[6]
                fmla v23.8h, v4.8h, v0.h[7]
                fmla v24.8h, v4.8h, v1.h[0]
                fmla v25.8h, v4.8h, v1.h[1]
                fmla v26.8h, v4.8h, v1.h[2]
                fmla v27.8h, v4.8h, v1.h[3]
                fmla v28.8h, v4.8h, v1.h[4]
                fmla v29.8h, v4.8h, v1.h[5]
                fmla v30.8h, v4.8h, v1.h[6]
                fmla v31.8h, v4.8h, v1.h[7]

                bne L16LoopZ

            L16LoopZEnd:

            mov dst_tmp, dst

            ld1 {v0.8h},  [dst_tmp], dst_w_step
            ld1 {v1.8h},  [dst_tmp], dst_w_step
            ld1 {v2.8h},  [dst_tmp], dst_w_step
            ld1 {v3.8h},  [dst_tmp], dst_w_step
            ld1 {v4.8h},  [dst_tmp], dst_w_step
            ld1 {v5.8h},  [dst_tmp], dst_w_step
            ld1 {v6.8h},  [dst_tmp], dst_w_step
            ld1 {v7.8h},  [dst_tmp], dst_w_step
            ld1 {v8.8h},  [dst_tmp], dst_w_step
            ld1 {v9.8h},  [dst_tmp], dst_w_step
            ld1 {v10.8h}, [dst_tmp], dst_w_step
            ld1 {v11.8h}, [dst_tmp], dst_w_step
            ld1 {v12.8h}, [dst_tmp], dst_w_step
            ld1 {v13.8h}, [dst_tmp], dst_w_step
            ld1 {v14.8h}, [dst_tmp], dst_w_step
            ld1 {v15.8h}, [dst_tmp], dst_w_step

            // add with stride
            fadd v16.8h, v16.8h, v0.8h
            fadd v17.8h, v17.8h, v1.8h
            fadd v18.8h, v18.8h, v2.8h
            fadd v19.8h, v19.8h, v3.8h
            fadd v20.8h, v20.8h, v4.8h
            fadd v21.8h, v21.8h, v5.8h
            fadd v22.8h, v22.8h, v6.8h
            fadd v23.8h, v23.8h, v7.8h
            fadd v24.8h, v24.8h, v8.8h
            fadd v25.8h, v25.8h, v9.8h
            fadd v26.8h, v26.8h, v10.8h
            fadd v27.8h, v27.8h, v11.8h
            fadd v28.8h, v28.8h, v12.8h
            fadd v29.8h, v29.8h, v13.8h
            fadd v30.8h, v30.8h, v14.8h
            fadd v31.8h, v31.8h, v15.8h

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
            st1 {v28.8h}, [dst], dst_w_step
            st1 {v29.8h}, [dst], dst_w_step
            st1 {v30.8h}, [dst], dst_w_step
            st1 {v31.8h}, [dst], dst_w_step

            sub dst, dst, x14
            add dst, dst, dilate_x_step

            mov input_c, x13
            subs fw, fw, #1
            sub x1, x1, x14
            mov src, x11
            bne L16LoopFX
        subs fh, fh, #1
        mov fw, x20
        mul x20, fw, dilate_x_step
        sub dst, dst, x20
        add dst, dst, dilate_y_step
        bne L16LoopFY

    mov fh, x19
    mul x20, fh, dilate_y_step
    sub dst, dst, x20
    add src, src, #32
    add dst, dst, x14
    mov weight, x12
    sub width, width, #16
    cmp width, #16
    bge L16Loop

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
            mov x13, input_c
            ld1 {v4.8h}, [weight], #16
            ld1 {v0.4h}, [src], x6
            fmul v16.8h, v4.8h, v0.h[0]
            fmul v17.8h, v4.8h, v0.h[1]
            fmul v18.8h, v4.8h, v0.h[2]
            fmul v19.8h, v4.8h, v0.h[3]

            subs input_c, input_c, #1
            beq L4LoopZEnd
            L4LoopZ:
                ld1 {v0.4h}, [src], x6
                ld1 {v4.8h}, [weight], #16
                subs input_c, input_c, #1
                fmla v16.8h, v4.8h, v0.h[0]
                fmla v17.8h, v4.8h, v0.h[1]
                fmla v18.8h, v4.8h, v0.h[2]
                fmla v19.8h, v4.8h, v0.h[3]

                bne L4LoopZ

            L4LoopZEnd:
            mov dst_tmp, dst

            ld1 {v0.8h}, [dst_tmp], dst_w_step
            ld1 {v1.8h}, [dst_tmp], dst_w_step
            ld1 {v2.8h}, [dst_tmp], dst_w_step
            ld1 {v3.8h}, [dst_tmp], dst_w_step

            // add with stride
            fadd v16.8h, v16.8h, v0.8h
            fadd v17.8h, v17.8h, v1.8h
            fadd v18.8h, v18.8h, v2.8h
            fadd v19.8h, v19.8h, v3.8h
            st1 {v16.8h}, [dst], dst_w_step
            st1 {v17.8h}, [dst], dst_w_step
            st1 {v18.8h}, [dst], dst_w_step
            st1 {v19.8h}, [dst], dst_w_step

            sub dst, dst, x14
            add dst, dst, dilate_x_step

            mov input_c, x13
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
    add src, src, #8
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
            mov x13, input_c
            eor v16.16b, v16.16b, v16.16b
            L1LoopZ:
                ld1 {v4.8h}, [weight], #16
                ld1 {v0.h}[0], [src], x6

                fmla v16.8h, v4.8h, v0.h[0]

                subs input_c, input_c, #1
                bne L1LoopZ

            L1LoopZEnd:
            ld1 {v0.8h}, [dst]
            // add with stride
            fadd v16.8h, v16.8h, v0.8h
            st1 {v16.8h}, [dst], dst_w_step

            sub dst, dst, x14
            add dst, dst, dilate_x_step

            mov input_c, x13
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
    add src, src, #2
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
