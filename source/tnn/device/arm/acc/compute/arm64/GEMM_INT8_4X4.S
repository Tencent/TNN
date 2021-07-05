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
asm_function GemmInt8Unit4x4 
//void GemmInt8Unit4x4(int8_t* src, const int8_t* weight, int8_t* dst, int src_w_step, int dst_depth, 
//     int cdiv8, float *scale, int32_t*bias, long relu, const int8_t* add_input, const float* add_scale, const int8_t* relu6_max)
//x0(src),
//x1(weight),
//x2(dst),
//x3(src_w_step),
//x4(dst_depth),
//x5(cdiv8),
//x6(scale),
//x7(bias)
//stack(relu)      [sp, 0]
//stack(add_input) [sp, 8]
//stack(add_scale) [sp, 16]
//stack(relu6_max) [sp, 24]

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

//prefetch data
//assume buffer c>=16, even c==8
ld1 {v12.16b, v13.16b}, [x1], #32 
add x11, x0, x3 
add x12, x11, x3 
add x13, x12, x3 

cmp x5, #2
ld1 {v14.16b, v15.16b}, [x1], #32 
ld1 {v8.16b}, [x0], #16 
ld1 {v9.16b}, [x11], #16 

blt C8First 

C16Start:
     sub x5, x5, #2 
     
     smull v0.8h, v12.8b, v8.8b 
     smull v1.8h, v13.8b, v8.8b 
     smlal2 v0.8h, v12.16b, v8.16b 
     smlal2 v1.8h, v13.16b, v8.16b 
     saddlp v16.4s, v0.8h 
     saddlp v17.4s, v1.8h 
      
     smull v2.8h, v14.8b, v8.8b 
     smull v3.8h, v15.8b, v8.8b 
     smull v4.8h, v12.8b, v9.8b 
     ld1 {v10.16b}, [x12], #16 
     smull v5.8h, v13.8b, v9.8b 
     smull v6.8h, v14.8b, v9.8b 
     smull v7.8h, v15.8b, v9.8b 
     smlal2 v2.8h, v14.16b, v8.16b 
     ld1 {v11.16b}, [x13], #16 
     smlal2 v3.8h, v15.16b, v8.16b 
     smlal2 v4.8h, v12.16b, v9.16b 
     smlal2 v5.8h, v13.16b, v9.16b 
     smlal2 v6.8h, v14.16b, v9.16b 
     smlal2 v7.8h, v15.16b, v9.16b 
     saddlp v18.4s, v2.8h 
     saddlp v19.4s, v3.8h 
     saddlp v20.4s, v4.8h 
     saddlp v21.4s, v5.8h 
     saddlp v22.4s, v6.8h 
     saddlp v23.4s, v7.8h 
     
     cmp x5, #2
     
     smull v0.8h, v12.8b, v10.8b 
     smull v1.8h, v13.8b, v10.8b 
     smull v2.8h, v14.8b, v10.8b 
     smull v3.8h, v15.8b, v10.8b 
     smlal2 v0.8h, v12.16b, v10.16b 
     smlal2 v1.8h, v13.16b, v10.16b 
     smlal2 v2.8h, v14.16b, v10.16b 
     smlal2 v3.8h, v15.16b, v10.16b 
         ld1 {v8.16b}, [x0], #16 
     saddlp v24.4s, v0.8h 
     saddlp v25.4s, v1.8h 
         ld1 {v9.16b}, [x11], #16 
     saddlp v26.4s, v2.8h 
     saddlp v27.4s, v3.8h 
      
     smull v4.8h, v12.8b, v11.8b 
     smull v5.8h, v13.8b, v11.8b 
     smull v6.8h, v14.8b, v11.8b 
     smull v7.8h, v15.8b, v11.8b 
     smlal2 v4.8h, v12.16b, v11.16b 
     smlal2 v5.8h, v13.16b, v11.16b 
     ld1 {v12.16b, v13.16b}, [x1], #32 
     smlal2 v6.8h, v14.16b, v11.16b 
     smlal2 v7.8h, v15.16b, v11.16b 
     saddlp v28.4s, v4.8h 
     saddlp v29.4s, v5.8h 
         ld1 {v14.16b, v15.16b}, [x1], #32 
     saddlp v30.4s, v6.8h 
     saddlp v31.4s, v7.8h 
      
     blt C8Last 
      
     C16Loop: 
         smull v0.8h, v12.8b, v8.8b 
         ld1 {v10.16b}, [x12], #16 
         smull v1.8h, v13.8b, v8.8b 
         smull v2.8h, v14.8b, v8.8b 
         smull v3.8h, v15.8b, v8.8b 
         smlal2 v0.8h, v12.16b, v8.16b 
         ld1 {v11.16b}, [x13], #16 
         smlal2 v1.8h, v13.16b, v8.16b 
         smlal2 v2.8h, v14.16b, v8.16b 
         smlal2 v3.8h, v15.16b, v8.16b 
         sadalp v16.4s, v0.8h 
         smull v4.8h, v12.8b, v9.8b 
         sadalp v17.4s, v1.8h 
         smull v5.8h, v13.8b, v9.8b 
         sadalp v18.4s, v2.8h 
         smull v6.8h, v14.8b, v9.8b 
         sadalp v19.4s, v3.8h 
         smull v7.8h, v15.8b, v9.8b 
      
         smlal2 v4.8h, v12.16b, v9.16b 
         ld1 {v8.16b}, [x0], #16 
         smlal2 v5.8h, v13.16b, v9.16b 
         smlal2 v6.8h, v14.16b, v9.16b 
         sub x5, x5, #2 
         smlal2 v7.8h, v15.16b, v9.16b 
         sadalp v20.4s, v4.8h 
         ld1 {v9.16b}, [x11], #16 
         smull v0.8h, v12.8b, v10.8b 
         sadalp v21.4s, v5.8h 
         smull v1.8h, v13.8b, v10.8b 
         sadalp v22.4s, v6.8h 
         smull v2.8h, v14.8b, v10.8b 
         sadalp v23.4s, v7.8h 
         smull v3.8h, v15.8b, v10.8b 
      
         smlal2 v0.8h, v12.16b, v10.16b 
         smlal2 v1.8h, v13.16b, v10.16b 
         smlal2 v2.8h, v14.16b, v10.16b 
         smlal2 v3.8h, v15.16b, v10.16b 
         sadalp v24.4s, v0.8h 
         smull v4.8h, v12.8b, v11.8b 
         sadalp v25.4s, v1.8h 
         smull v5.8h, v13.8b, v11.8b 
         sadalp v26.4s, v2.8h 
         smull v6.8h, v14.8b, v11.8b 
         sadalp v27.4s, v3.8h 
         smull v7.8h, v15.8b, v11.8b 
      
         smlal2 v4.8h, v12.16b, v11.16b 
         smlal2 v5.8h, v13.16b, v11.16b 
         cmp x5, #2 
         smlal2 v6.8h, v14.16b, v11.16b 
         ld1 {v12.16b, v13.16b}, [x1], #32 
         smlal2 v7.8h, v15.16b, v11.16b 
         sadalp v28.4s, v4.8h 
         ld1 {v14.16b, v15.16b}, [x1], #32 
         sadalp v29.4s, v5.8h 
         sadalp v30.4s, v6.8h 
         sadalp v31.4s, v7.8h 
         bge C16Loop 
 
C8Last:
     cmp x5, #1
     blt LoopEnd 
     smull v0.8h, v12.8b, v8.8b 
     ld1 {v10.16b}, [x12], #16 
     smull v1.8h, v13.8b, v8.8b 
     smull v2.8h, v14.8b, v8.8b 
     smull v3.8h, v15.8b, v8.8b 
     ld1 {v11.16b}, [x13], #16 
     sadalp v16.4s, v0.8h 
     smull v4.8h, v12.8b, v9.8b 
     sadalp v17.4s, v1.8h 
     smull v5.8h, v13.8b, v9.8b 
     sadalp v18.4s, v2.8h 
     smull v6.8h, v14.8b, v9.8b 
     sadalp v19.4s, v3.8h 
     smull v7.8h, v15.8b, v9.8b 
     
     sadalp v20.4s, v4.8h 
     smull v0.8h, v12.8b, v10.8b 
     sadalp v21.4s, v5.8h 
     smull v1.8h, v13.8b, v10.8b 
     sadalp v22.4s, v6.8h 
     smull v2.8h, v14.8b, v10.8b 
     sadalp v23.4s, v7.8h 
     smull v3.8h, v15.8b, v10.8b 
     
     sadalp v24.4s, v0.8h 
     smull v4.8h, v12.8b, v11.8b 
     sadalp v25.4s, v1.8h 
     smull v5.8h, v13.8b, v11.8b 
     sadalp v26.4s, v2.8h 
     smull v6.8h, v14.8b, v11.8b 
     sadalp v27.4s, v3.8h 
     smull v7.8h, v15.8b, v11.8b 
     
     sadalp v28.4s, v4.8h 
     sadalp v29.4s, v5.8h 
     sadalp v30.4s, v6.8h 
     sadalp v31.4s, v7.8h 
     b LoopEnd
      
C8First:
     cmp x5, #1
     blt LoopEnd 
     smull v0.8h, v12.8b, v8.8b 
     ld1 {v10.16b}, [x12], #16 
     smull v1.8h, v13.8b, v8.8b 
     smull v2.8h, v14.8b, v8.8b 
     smull v3.8h, v15.8b, v8.8b 
     ld1 {v11.16b}, [x13], #16 
     saddlp v16.4s, v0.8h 
     smull v4.8h, v12.8b, v9.8b 
     saddlp v17.4s, v1.8h 
     smull v5.8h, v13.8b, v9.8b 
     saddlp v18.4s, v2.8h 
     smull v6.8h, v14.8b, v9.8b 
     saddlp v19.4s, v3.8h 
     smull v7.8h, v15.8b, v9.8b 
     
     saddlp v20.4s, v4.8h 
     smull v0.8h, v12.8b, v10.8b 
     saddlp v21.4s, v5.8h 
     smull v1.8h, v13.8b, v10.8b 
     saddlp v22.4s, v6.8h 
     smull v2.8h, v14.8b, v10.8b 
     saddlp v23.4s, v7.8h 
     smull v3.8h, v15.8b, v10.8b 
     
     saddlp v24.4s, v0.8h 
     smull v4.8h, v12.8b, v11.8b 
     saddlp v25.4s, v1.8h 
     smull v5.8h, v13.8b, v11.8b 
     saddlp v26.4s, v2.8h 
     smull v6.8h, v14.8b, v11.8b 
     saddlp v27.4s, v3.8h 
     smull v7.8h, v15.8b, v11.8b 
     
     saddlp v28.4s, v4.8h 
     saddlp v29.4s, v5.8h 
     saddlp v30.4s, v6.8h 
     saddlp v31.4s, v7.8h 
LoopEnd: 
      
     ld1 {v0.4s}, [x7], #16 
     addp v4.4s, v16.4s, v17.4s 
     addp v5.4s, v18.4s, v19.4s 
     addp v6.4s, v20.4s, v21.4s 
     addp v7.4s, v22.4s, v23.4s 
     addp v8.4s, v24.4s, v25.4s 
     addp v9.4s, v26.4s, v27.4s 
     addp v10.4s, v28.4s, v29.4s 
     addp v11.4s, v30.4s, v31.4s 
      
     addp v12.4s, v4.4s, v5.4s 
     addp v13.4s, v6.4s, v7.4s 
     addp v14.4s, v8.4s, v9.4s 
     addp v15.4s, v10.4s, v11.4s 
     ld1 {v1.4s}, [x6], #16 
     add v16.4s, v12.4s, v0.4s 
     add v17.4s, v13.4s, v0.4s 
     add v18.4s, v14.4s, v0.4s 
     add v19.4s, v15.4s, v0.4s 
      
     scvtf v4.4s, v16.4s 
     scvtf v5.4s, v17.4s 
     scvtf v6.4s, v18.4s 
     scvtf v7.4s, v19.4s 
      
     fmul v12.4s, v4.4s, v1.4s  // result = result * scale
     fmul v13.4s, v5.4s, v1.4s 
      
     fmul v14.4s, v6.4s, v1.4s 
     fmul v15.4s, v7.4s, v1.4s 

     ldr x9, [sp, #0]    // relu
     ldr x7, [sp, #8]    // add_input
     ldr x8, [sp, #16]   // add_scale

     cmp x9, #-1         // relu (conv - relu - add, relu == -1)
     bne Add
     movi v3.16b, #0
     fmax v12.4s, v12.4s, v3.4s
     fmax v13.4s, v13.4s, v3.4s
     fmax v14.4s, v14.4s, v3.4s
     fmax v15.4s, v15.4s, v3.4s

Add:
     cbz x7, Relu        // if add_input == 0, skip
     ld1 {v28.s}[0], [x7], x4
     ld1 {v28.s}[1], [x7], x4
     ld1 {v29.s}[0], [x7], x4
     ld1 {v29.s}[1], [x7]
     ld1 {v30.4s}, [x8]

     sxtl  v26.8h, v28.8b
     sxtl  v27.8h, v29.8b
     sxtl  v20.4s, v26.4h
     sxtl2 v21.4s, v26.8h
     sxtl  v22.4s, v27.4h
     sxtl2 v23.4s, v27.8h

     scvtf v24.4s, v20.4s
     scvtf v25.4s, v21.4s
     scvtf v26.4s, v22.4s
     scvtf v27.4s, v23.4s

     fmla v12.4s, v24.4s, v30.4s  // result += add_input * add_scale
     fmla v13.4s, v25.4s, v30.4s
     fmla v14.4s, v26.4s, v30.4s
     fmla v15.4s, v27.4s, v30.4s

Relu:
     fcvtas v8.4s, v12.4s 
     fcvtas v9.4s, v13.4s 
     fcvtas v10.4s, v14.4s 
     fcvtas v11.4s, v15.4s 
      
     sqxtn v0.4h, v8.4s 
     sqxtn v1.4h, v10.4s 
     sqxtn2 v0.8h, v9.4s 
     sqxtn2 v1.8h, v11.4s 
     sqxtn v2.8b, v0.8h 
     sqxtn v4.8b, v1.8h

     cmp x9, #1         // relu (conv add relu, relu == 1 or relu6 == 2)
     blt Store
     movi v3.8b, #0
     smax v2.8b, v2.8b, v3.8b
     smax v4.8b, v4.8b, v3.8b

     cmp x9, #2         // relu6
     bne Store
     ldr x8, [sp, #24]  // relu6_max
     ld1r {v3.2s}, [x8]
     smin v2.8b, v2.8b, v3.8b
     smin v4.8b, v4.8b, v3.8b

Store:
     st1 {v2.s}[0], [x2], x4
     st1 {v2.s}[1], [x2], x4

     st1 {v4.s}[0], [x2], x4
     st1 {v4.s}[1], [x2]

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
