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



#endif
#endif