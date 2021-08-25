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

asm_function DepthwiseI8K3S1Kernel
//void DepthwiseI8K3S1Kernel(int8_t *dst_z,
//                        const int8_t *src,
//                        const int8_t* weight_z,
//                        const int32_t* bias_z,
//                        const float* scale_z,
//                        long src_y_step,
//                        long dst_depth,
//                        long width)
//x0(dst_z),
//x1(src),
//x2(weight_z),
//x3(bias_z),
//x4(scale_z),
//x5(src_y_step),
//x6(dst_depth),
//x7(width)

.macro COMPUTE_SMLAL_UNIT z0 z1 z2 z3 y
    smlal  v20.4s, \z0\().4h, \y\().4h
    smlal2 v21.4s, \z0\().8h, \y\().8h
    smlal  v22.4s, \z1\().4h, \y\().4h
    smlal2 v23.4s, \z1\().8h, \y\().8h
    smlal  v24.4s, \z2\().4h, \y\().4h
    smlal2 v25.4s, \z2\().8h, \y\().8h
    smlal  v26.4s, \z3\().4h, \y\().4h
    smlal2 v27.4s, \z3\().8h, \y\().8h
.endm

cmp x7, #0
ble End

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

mov x9, #3
mul x9, x6, x9
add x10, x2, x9
add x11, x2, x9, lsl#1

// weight
ld1 {v0.8b}, [x2], x6
ld1 {v1.8b}, [x2], x6
ld1 {v2.8b}, [x2]
ld1 {v3.8b}, [x10], x6
ld1 {v4.8b}, [x10], x6
ld1 {v5.8b}, [x10]
ld1 {v6.8b}, [x11], x6
ld1 {v7.8b}, [x11], x6
ld1 {v8.8b}, [x11]

sxtl v0.8h, v0.8b
sxtl v1.8h, v1.8b
sxtl v2.8h, v2.8b
sxtl v3.8h, v3.8b
sxtl v4.8h, v4.8b
sxtl v5.8h, v5.8b
sxtl v6.8h, v6.8b
sxtl v7.8h, v7.8b
sxtl v8.8h, v8.8b

// bias
ldr q9, [x3]
ldr q10, [x3, #16]

// scale
ldr q11, [x4]
ldr q12, [x4, #16]

LoopDw:
    cmp x7, #3
    ble LoopDwEnd

    sub x7, x7, #4

    mov x9, x1
    ld1 {v13.8b}, [x9], x6
    ld1 {v14.8b}, [x9], x6
    ld1 {v15.8b}, [x9], x6
    ld1 {v16.8b}, [x9], x6
    ld1 {v17.8b}, [x9], x6
    ld1 {v18.8b}, [x9]
    add x9, x1, x5
    add x10, x1, x5, lsl#1

    mov v20.16b, v9.16b
    mov v21.16b, v10.16b
    mov v22.16b, v9.16b
    mov v23.16b, v10.16b
    mov v24.16b, v9.16b
    mov v25.16b, v10.16b
    mov v26.16b, v9.16b
    mov v27.16b, v10.16b

    sxtl v13.8h, v13.8b
    sxtl v14.8h, v14.8b
    sxtl v15.8h, v15.8b
    sxtl v16.8h, v16.8b
    sxtl v17.8h, v17.8b
    sxtl v18.8h, v18.8b

    ld1 {v19.8b}, [x9], x6
    ld1 {v28.8b}, [x9], x6
    ld1 {v29.8b}, [x9], x6
    ld1 {v30.8b}, [x9], x6
    ld1 {v31.8b}, [x9], x6
    COMPUTE_SMLAL_UNIT v13, v14, v15, v16, v0
    ld1 {v13.8b}, [x9]
    COMPUTE_SMLAL_UNIT v14, v15, v16, v17, v1
    COMPUTE_SMLAL_UNIT v15, v16, v17, v18, v2

    sxtl v19.8h, v19.8b
    sxtl v28.8h, v28.8b
    sxtl v29.8h, v29.8b
    sxtl v30.8h, v30.8b
    sxtl v31.8h, v31.8b
    sxtl v13.8h, v13.8b

    ld1 {v14.8b}, [x10], x6
    ld1 {v15.8b}, [x10], x6
    ld1 {v16.8b}, [x10], x6
    ld1 {v17.8b}, [x10], x6
    ld1 {v18.8b}, [x10], x6
    COMPUTE_SMLAL_UNIT v19, v28, v29, v30, v3
    ld1 {v19.8b}, [x10]
    COMPUTE_SMLAL_UNIT v28, v29, v30, v31, v4
    COMPUTE_SMLAL_UNIT v29, v30, v31, v13, v5

    sxtl v14.8h, v14.8b
    sxtl v15.8h, v15.8b
    sxtl v16.8h, v16.8b
    sxtl v17.8h, v17.8b
    sxtl v18.8h, v18.8b
    sxtl v19.8h, v19.8b

    COMPUTE_SMLAL_UNIT v14, v15, v16, v17, v6
    COMPUTE_SMLAL_UNIT v15, v16, v17, v18, v7
    COMPUTE_SMLAL_UNIT v16, v17, v18, v19, v8

    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    scvtf v24.4s, v24.4s
    scvtf v25.4s, v25.4s
    scvtf v26.4s, v26.4s
    scvtf v27.4s, v27.4s

    fmul v20.4s, v20.4s, v11.4s
    fmul v21.4s, v21.4s, v12.4s
    fmul v22.4s, v22.4s, v11.4s
    fmul v23.4s, v23.4s, v12.4s
    fmul v24.4s, v24.4s, v11.4s
    fmul v25.4s, v25.4s, v12.4s
    fmul v26.4s, v26.4s, v11.4s
    fmul v27.4s, v27.4s, v12.4s

    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s
    fcvtas v22.4s, v22.4s
    fcvtas v23.4s, v23.4s
    fcvtas v24.4s, v24.4s
    fcvtas v25.4s, v25.4s
    fcvtas v26.4s, v26.4s
    fcvtas v27.4s, v27.4s

    sqxtn  v20.4h, v20.4s
    sqxtn  v22.4h, v22.4s
    sqxtn  v24.4h, v24.4s
    sqxtn  v26.4h, v26.4s
    sqxtn2 v20.8h, v21.4s
    sqxtn2 v22.8h, v23.4s
    sqxtn2 v24.8h, v25.4s
    sqxtn2 v26.8h, v27.4s

    sqxtn v20.8b, v20.8h
    sqxtn v22.8b, v22.8h
    sqxtn v24.8b, v24.8h
    sqxtn v26.8b, v26.8h

    st1 {v20.8b}, [x0], x6
    st1 {v22.8b}, [x0], x6
    st1 {v24.8b}, [x0], x6
    st1 {v26.8b}, [x0], x6

    // src += 4 * dst_depth
    add x1, x1, x6, lsl#2

    b LoopDw

LoopDwEnd:

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

End:

ret

#endif
