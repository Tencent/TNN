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

asm_function GemvInt8Sdot
//void GemvInt8Sdot(int8_t* dst, const int8_t* src, const int8_t* weight,
//                  const int32_t* bias, const float* scale, long ic_r4, long oc_r4)
//x0(dst),
//x1(src),
//x2(weight),
//x3(bias),
//x4(scale),
//x5(ic_r4),
//x6(oc_r4),

LoopOc16:
    cmp x6, #15
    ble LoopOc4

    sub x6, x6, #16
    ldr q0, [x3]
    ldr q1, [x3, #16]
    ldr q2, [x3, #32]
    ldr q3, [x3, #48]
    add x3, x3, #64

    mov x9, x1              // src_ptr 
    mov x10, x5             // ic counter
    mov x11, x2             // weight_ptr

    Oc16L16:
        cmp x10, #15
        ble Oc16L8

        sub x10, x10, #16
        ld1 {v4.16b}, [x9], #16
        ldr q16, [x11]
        ldr q17, [x11, #16]
        ldr q18, [x11, #32]
        ldr q19, [x11, #48]
        ldr q20, [x11, #64]
        ldr q21, [x11, #80]
        ldr q22, [x11, #96]
        ldr q23, [x11, #112]
        ldr q24, [x11, #128]
        ldr q25, [x11, #144]
        ldr q26, [x11, #160]
        ldr q27, [x11, #176]
        ldr q28, [x11, #192]
        ldr q29, [x11, #208]
        ldr q30, [x11, #224]
        ldr q31, [x11, #240]
        add x11, x11, #256

        .word 0x4f84e200 // sdot v0.4s, v16.16b, v4.4b[0]
        .word 0x4f84e221 // sdot v1.4s, v17.16b, v4.4b[0]
        .word 0x4f84e242 // sdot v2.4s, v18.16b, v4.4b[0]
        .word 0x4f84e263 // sdot v3.4s, v19.16b, v4.4b[0]
        .word 0x4fa4e280 // sdot v0.4s, v20.16b, v4.4b[1]
        .word 0x4fa4e2a1 // sdot v1.4s, v21.16b, v4.4b[1]
        .word 0x4fa4e2c2 // sdot v2.4s, v22.16b, v4.4b[1]
        .word 0x4fa4e2e3 // sdot v3.4s, v23.16b, v4.4b[1]
        .word 0x4f84eb00 // sdot v0.4s, v24.16b, v4.4b[2]
        .word 0x4f84eb21 // sdot v1.4s, v25.16b, v4.4b[2]
        .word 0x4f84eb42 // sdot v2.4s, v26.16b, v4.4b[2]
        .word 0x4f84eb63 // sdot v3.4s, v27.16b, v4.4b[2]
        .word 0x4fa4eb80 // sdot v0.4s, v28.16b, v4.4b[3]
        .word 0x4fa4eba1 // sdot v1.4s, v29.16b, v4.4b[3]
        .word 0x4fa4ebc2 // sdot v2.4s, v30.16b, v4.4b[3]
        .word 0x4fa4ebe3 // sdot v3.4s, v31.16b, v4.4b[3]
        b Oc16L16

    Oc16L8:
        cmp x10, #7
        ble Oc16L4

        sub x10, x10, #8
        ld1 {v4.8b}, [x9], #8
        ldr q16, [x11]
        ldr q17, [x11, #16]
        ldr q18, [x11, #32]
        ldr q19, [x11, #48]
        ldr q20, [x11, #64]
        ldr q21, [x11, #80]
        ldr q22, [x11, #96]
        ldr q23, [x11, #112]
        add x11, x11, #128

        .word 0x4f84e200 // sdot v0.4s, v16.16b, v4.4b[0]
        .word 0x4f84e221 // sdot v1.4s, v17.16b, v4.4b[0]
        .word 0x4f84e242 // sdot v2.4s, v18.16b, v4.4b[0]
        .word 0x4f84e263 // sdot v3.4s, v19.16b, v4.4b[0]
        .word 0x4fa4e280 // sdot v0.4s, v20.16b, v4.4b[1]
        .word 0x4fa4e2a1 // sdot v1.4s, v21.16b, v4.4b[1]
        .word 0x4fa4e2c2 // sdot v2.4s, v22.16b, v4.4b[1]
        .word 0x4fa4e2e3 // sdot v3.4s, v23.16b, v4.4b[1]
        b Oc16L8
    
    Oc16L4:
        cmp x10, #3
        ble Oc16L4End

        sub x10, x10, #4
        ld1 {v4.s}[0], [x9], #4
        ldr q16, [x11]
        ldr q17, [x11, #16]
        ldr q18, [x11, #32]
        ldr q19, [x11, #48]
        add x11, x11, #64

        .word 0x4f84e200 // sdot v0.4s, v16.16b, v4.4b[0]
        .word 0x4f84e221 // sdot v1.4s, v17.16b, v4.4b[0]
        .word 0x4f84e242 // sdot v2.4s, v18.16b, v4.4b[0]
        .word 0x4f84e263 // sdot v3.4s, v19.16b, v4.4b[0]
        b Oc16L4

    Oc16L4End:

        ldr q16, [x4]
        ldr q17, [x4, #16]
        ldr q18, [x4, #32]
        ldr q19, [x4, #48]
        add x4, x4, #64

        scvtf v0.4s, v0.4s
        scvtf v1.4s, v1.4s
        scvtf v2.4s, v2.4s
        scvtf v3.4s, v3.4s

        fmul v0.4s, v0.4s, v16.4s
        fmul v1.4s, v1.4s, v17.4s
        fmul v2.4s, v2.4s, v18.4s
        fmul v3.4s, v3.4s, v19.4s

        fcvtas v0.4s, v0.4s
        fcvtas v1.4s, v1.4s
        fcvtas v2.4s, v2.4s
        fcvtas v3.4s, v3.4s

        sqxtn  v0.4h, v0.4s
        sqxtn2 v0.8h, v1.4s
        sqxtn  v1.4h, v2.4s
        sqxtn2 v1.8h, v3.4s
        sqxtn  v0.8b, v0.8h
        sqxtn  v1.8b, v1.8h

        st1 {v0.8b, v1.8b}, [x0], #16
        // weight += 16 * ic_r4
        add x2, x2, x5, lsl#4

        b LoopOc16

LoopOc4:
    cmp x6, #3
    ble END

    sub x6, x6, #4
    ld1 {v0.16b}, [x3], #16

    mov x9, x1              // src_ptr 
    mov x10, x5             // ic counter
    mov x11, x2             // weight_ptr

    Oc4L16:
        cmp x10, #15
        ble Oc4L4

        sub x10, x10, #16
        ld1 {v4.16b}, [x9], #16
        ldr q16, [x11]
        ldr q17, [x11, #16]
        ldr q18, [x11, #32]
        ldr q19, [x11, #48]
        add x11, x11, #64

        .word 0x4f84e200 // sdot v0.4s, v16.16b, v4.4b[0]
        .word 0x4fa4e220 // sdot v0.4s, v17.16b, v4.4b[1]
        .word 0x4f84ea40 // sdot v0.4s, v18.16b, v4.4b[2]
        .word 0x4fa4ea60 // sdot v0.4s, v19.16b, v4.4b[3]

        b Oc4L16

    Oc4L4:
        cmp x10, #3
        ble Oc4L4End

        sub x10, x10, #4
        ld1 {v4.s}[0], [x9], #4
        ld1 {v16.16b}, [x11], #16

        .word 0x4f84e200 // sdot v0.4s, v16.16b, v4.4b[0]

        b Oc4L4

    Oc4L4End:

        ld1 {v16.4s}, [x4], #16
        scvtf v0.4s, v0.4s
        fmul v0.4s, v0.4s, v16.4s
        fcvtas v0.4s, v0.4s
        sqxtn  v0.4h, v0.4s
        sqxtn  v0.8b, v0.8h

        st1 {v0.s}[0], [x0], #4
        // weight += 4 * ic_r4
        add x2, x2, x5, lsl#2

        b LoopOc4

END:

ret

#endif
#endif
