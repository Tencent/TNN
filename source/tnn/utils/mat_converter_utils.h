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

#ifndef TNN_UTILS_MAT_CONVERTER_UTILS_H_
#define TNN_UTILS_MAT_CONVERTER_UTILS_H_

#include <cmath>

#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/core/mat.h"

namespace TNN_NS {

Status CheckMatConverterParams(Mat& src, Mat& dst, bool check_same_device);

void CalculatePositionAndRatio(int length, double scale, int border, int channel,
                                         int* position, short* ratio);

void CalculatePositionAndMask(int length, double scale, int border, int channel,
                                     int* position, uint8_t* mask);

// Meanings of xofs, yofs, ialpha, ibeta in src image:
//                               |  ialpha[2*x]  |  ialpha[2*x+1]  |
//     --       (xofs[x], yofs[y])                                 (xofs[x]+1, yofs[y])
// ibeta[2*y]
//     --                              (x*scale_x, y*scale_y)
// ibeta[2*y+1]
//     --       (xofs[x], yofs[y]+1)                               (xofs[x]+1, yofs[y]+1)
void GetResizeBuf(int src_w, int src_h, int w, int h, int c, int** buf);


// Meanings of xofs, yofs, ialpha, ibeta in src image:
//                               |  ialpha[x] (1: left, 0: right)  |
//     --       (xofs[x], yofs[y])                                 (xofs[x]+1, yofs[y])
// ibeta[y]
// (1: top,                            (x*scale_x, y*scale_y)
//  0: bottom)
//     --       (xofs[x], yofs[y]+1)                               (xofs[x]+1, yofs[y]+1)
void GetResizeBufNearset(int src_w, int src_h, int w, int h, int c, int** buf);

void InitInterTab1D(float* tab, int tabsz);

void WarpAffineMatrixInverse(const float (*transform)[3], double* inverse);

int GetMatElementSize(Mat* mat);

}  // namespace TNN_NS

#endif
