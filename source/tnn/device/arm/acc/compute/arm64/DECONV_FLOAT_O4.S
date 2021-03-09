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

asm_function DeconvFloatO4
//void DeconvFloatO4(float* dst,           // x0
//                     const float* src,     // x1
//                     const float* weight,  // x2
//                     int width,            // x3
//                     int dst_w_step,       // x4
//                     int src_depth_quad,   // x5
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
ic4          .req x5
fw           .req x7
fh           .req x8
dilate_x_step .req x9
dilate_y_step .req x10

//Auto Load:
//x0:dst, x1:src, x2:weight, x3:width, x4:src_w_step, x5:src_depth_quad, x6: src_depth_step, x7:fw

//Load from sp
//x8:fh, x9:dilate_x_step, x10:dilate_y_step
// eor x8, x8, x8
ldr x8, [sp, #0]
// eor x9, x9, x9
ldr x9, [sp, #8]
// eor x10, x10, x10   
ldr x10, [sp, #16]

//step multi by sizeof(float)
mov x12, #4
mul x10, x12, x10
mul x9, x12, x9
mul x6, x12, x6
mul x4, x12, x4

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
            mov x13, ic4
            ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [weight], #64
            ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
            ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
            ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x1], #64
            ld1 {v12.4s, v13.4s}, [x1], #32

            fmul v14.4s, v28.4s, v0.s[0]
            fmul v15.4s, v28.4s, v1.s[0]
            fmul v16.4s, v28.4s, v2.s[0]
            fmul v17.4s, v28.4s, v3.s[0]
            fmul v18.4s, v28.4s, v4.s[0]
            fmul v19.4s, v28.4s, v5.s[0]
            fmul v20.4s, v28.4s, v6.s[0]
            fmul v21.4s, v28.4s, v7.s[0]
            fmul v22.4s, v28.4s, v8.s[0]
            fmul v23.4s, v28.4s, v9.s[0]
            fmul v24.4s, v28.4s, v10.s[0]
            fmul v25.4s, v28.4s, v11.s[0]
            fmul v26.4s, v28.4s, v12.s[0]
            fmul v27.4s, v28.4s, v13.s[0]

            subs ic4, ic4, #1
            beq L14LoopZEnd
            L14LoopZ:
                sub src, src, #224
                add src, src, x6

                fmla v14.4s, v29.4s, v0.s[1]
                fmla v15.4s, v29.4s, v1.s[1]
                fmla v16.4s, v29.4s, v2.s[1]
                fmla v17.4s, v29.4s, v3.s[1]
                fmla v18.4s, v29.4s, v4.s[1]
                fmla v19.4s, v29.4s, v5.s[1]
                fmla v20.4s, v29.4s, v6.s[1]
                fmla v21.4s, v29.4s, v7.s[1]
                fmla v22.4s, v29.4s, v8.s[1]
                fmla v23.4s, v29.4s, v9.s[1]
                fmla v24.4s, v29.4s, v10.s[1]
                fmla v25.4s, v29.4s, v11.s[1]
                fmla v26.4s, v29.4s, v12.s[1]
                fmla v27.4s, v29.4s, v13.s[1]

                fmla v14.4s, v30.4s, v0.s[2]
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

                fmla v14.4s, v31.4s, v0.s[3]
                fmla v15.4s, v31.4s, v1.s[3]
                fmla v16.4s, v31.4s, v2.s[3]
                fmla v17.4s, v31.4s, v3.s[3]
                fmla v18.4s, v31.4s, v4.s[3]
                fmla v19.4s, v31.4s, v5.s[3]
                fmla v20.4s, v31.4s, v6.s[3]
                fmla v21.4s, v31.4s, v7.s[3]
                fmla v22.4s, v31.4s, v8.s[3]
                fmla v23.4s, v31.4s, v9.s[3]
                fmla v24.4s, v31.4s, v10.s[3]
                fmla v25.4s, v31.4s, v11.s[3]
                fmla v26.4s, v31.4s, v12.s[3]
                fmla v27.4s, v31.4s, v13.s[3]

                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [weight], #64
                ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
                ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
                ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x1], #64
                ld1 {v12.4s, v13.4s}, [x1], #32

                fmla v14.4s, v28.4s, v0.s[0]
                fmla v15.4s, v28.4s, v1.s[0]
                fmla v16.4s, v28.4s, v2.s[0]
                fmla v17.4s, v28.4s, v3.s[0]
                fmla v18.4s, v28.4s, v4.s[0]
                fmla v19.4s, v28.4s, v5.s[0]
                fmla v20.4s, v28.4s, v6.s[0]
                fmla v21.4s, v28.4s, v7.s[0]
                fmla v22.4s, v28.4s, v8.s[0]
                fmla v23.4s, v28.4s, v9.s[0]
                fmla v24.4s, v28.4s, v10.s[0]
                fmla v25.4s, v28.4s, v11.s[0]
                fmla v26.4s, v28.4s, v12.s[0]
                fmla v27.4s, v28.4s, v13.s[0]

                subs ic4, ic4, #1
                bne L14LoopZ

            L14LoopZEnd:
            fmla v14.4s, v29.4s, v0.s[1]
            fmla v15.4s, v29.4s, v1.s[1]
            fmla v16.4s, v29.4s, v2.s[1]
            fmla v17.4s, v29.4s, v3.s[1]
            fmla v18.4s, v29.4s, v4.s[1]
            fmla v19.4s, v29.4s, v5.s[1]
            fmla v20.4s, v29.4s, v6.s[1]
            fmla v21.4s, v29.4s, v7.s[1]
            fmla v22.4s, v29.4s, v8.s[1]
            fmla v23.4s, v29.4s, v9.s[1]
            fmla v24.4s, v29.4s, v10.s[1]
            fmla v25.4s, v29.4s, v11.s[1]
            fmla v26.4s, v29.4s, v12.s[1]
            fmla v27.4s, v29.4s, v13.s[1]

            fmla v14.4s, v30.4s, v0.s[2]
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

            fmla v14.4s, v31.4s, v0.s[3]
            fmla v15.4s, v31.4s, v1.s[3]
            fmla v16.4s, v31.4s, v2.s[3]
            fmla v17.4s, v31.4s, v3.s[3]
            fmla v18.4s, v31.4s, v4.s[3]
            fmla v19.4s, v31.4s, v5.s[3]
            fmla v20.4s, v31.4s, v6.s[3]
            fmla v21.4s, v31.4s, v7.s[3]
            fmla v22.4s, v31.4s, v8.s[3]
            fmla v23.4s, v31.4s, v9.s[3]
            fmla v24.4s, v31.4s, v10.s[3]
            fmla v25.4s, v31.4s, v11.s[3]
            fmla v26.4s, v31.4s, v12.s[3]
            fmla v27.4s, v31.4s, v13.s[3]            
            // add with stride
            ld1 {v0.4s}, [dst]
            fadd v14.4s, v14.4s, v0.4s
            st1 {v14.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v15.4s, v15.4s, v0.4s
            st1 {v15.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v16.4s, v16.4s, v0.4s
            st1 {v16.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v17.4s, v17.4s, v0.4s
            st1 {v17.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v18.4s, v18.4s, v0.4s
            st1 {v18.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v19.4s, v19.4s, v0.4s
            st1 {v19.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v20.4s, v20.4s, v0.4s
            st1 {v20.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v21.4s, v21.4s, v0.4s
            st1 {v21.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v22.4s, v22.4s, v0.4s
            st1 {v22.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v23.4s, v23.4s, v0.4s
            st1 {v23.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v24.4s, v24.4s, v0.4s
            st1 {v24.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v25.4s, v25.4s, v0.4s
            st1 {v25.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v26.4s, v26.4s, v0.4s
            st1 {v26.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v27.4s, v27.4s, v0.4s
            st1 {v27.4s}, [dst], dst_w_step

            sub dst, dst, x14
            add dst, dst, dilate_x_step

            mov ic4, x13
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
            mov x13, ic4
            ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [weight], #64
            ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
            fmul v14.4s, v28.4s, v0.s[0]
            fmul v15.4s, v28.4s, v1.s[0]
            fmul v16.4s, v28.4s, v2.s[0]
            fmul v17.4s, v28.4s, v3.s[0]

            subs ic4, ic4, #1
            beq L4LoopZEnd
            L4LoopZ:
                sub src, src, #64
                add src, src, x6

                fmla v14.4s, v29.4s, v0.s[1]
                fmla v15.4s, v29.4s, v1.s[1]
                fmla v16.4s, v29.4s, v2.s[1]
                fmla v17.4s, v29.4s, v3.s[1]

                fmla v14.4s, v30.4s, v0.s[2]
                fmla v15.4s, v30.4s, v1.s[2]
                fmla v16.4s, v30.4s, v2.s[2]
                fmla v17.4s, v30.4s, v3.s[2]

                fmla v14.4s, v31.4s, v0.s[3]
                fmla v15.4s, v31.4s, v1.s[3]
                fmla v16.4s, v31.4s, v2.s[3]
                fmla v17.4s, v31.4s, v3.s[3]

                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [weight], #64

                ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64 

                fmla v14.4s, v28.4s, v0.s[0]
                fmla v15.4s, v28.4s, v1.s[0]
                fmla v16.4s, v28.4s, v2.s[0]
                fmla v17.4s, v28.4s, v3.s[0]

                subs ic4, ic4, #1
                bne L4LoopZ

            L4LoopZEnd:
            fmla v14.4s, v29.4s, v0.s[1]
            fmla v15.4s, v29.4s, v1.s[1]
            fmla v16.4s, v29.4s, v2.s[1]
            fmla v17.4s, v29.4s, v3.s[1]

            fmla v14.4s, v30.4s, v0.s[2]
            fmla v15.4s, v30.4s, v1.s[2]
            fmla v16.4s, v30.4s, v2.s[2]
            fmla v17.4s, v30.4s, v3.s[2]

            fmla v14.4s, v31.4s, v0.s[3]
            fmla v15.4s, v31.4s, v1.s[3]
            fmla v16.4s, v31.4s, v2.s[3]
            fmla v17.4s, v31.4s, v3.s[3]
            // add with stride
            ld1 {v0.4s}, [dst]
            fadd v14.4s, v14.4s, v0.4s
            st1 {v14.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v15.4s, v15.4s, v0.4s
            st1 {v15.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v16.4s, v16.4s, v0.4s
            st1 {v16.4s}, [dst], dst_w_step

            ld1 {v0.4s}, [dst]
            fadd v17.4s, v17.4s, v0.4s
            st1 {v17.4s}, [dst], dst_w_step

            sub dst, dst, x14
            add dst, dst, dilate_x_step

            mov ic4, x13
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
            mov x13, ic4
            movi v14.4s, #0
            movi v15.4s, #0
            L1LoopZ:
                ld1 {v28.4s, v29.4s}, [weight], #32
                ld1 {v0.4s}, [x1], #16

                fmla v14.4s, v28.4s, v0.s[0]
                fmla v15.4s, v29.4s, v0.s[1]
                ld1 {v30.4s, v31.4s}, [weight], #32
                fmla v14.4s, v30.4s, v0.s[2]
                fmla v15.4s, v31.4s, v0.s[3]

                sub src, src, #16
                add src, src, x6

                subs ic4, ic4, #1
                bne L1LoopZ

            L1LoopZEnd:
            // add with stride
            fadd v14.4s, v14.4s, v15.4s
            ld1 {v0.4s}, [dst]
            fadd v14.4s, v14.4s, v0.4s
            st1 {v14.4s}, [dst], dst_w_step

            sub dst, dst, x14
            add dst, dst, dilate_x_step

            mov ic4, x13
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
