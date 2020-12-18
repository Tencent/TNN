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

#ifdef __arm__
#ifndef __aarch64__

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



#endif
#endif