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

#include "tnn/utils/mat_converter_utils.h"

#include <climits>
#include <algorithm>

namespace TNN_NS {

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX)

Status CheckMatConverterParams(Mat& src, Mat& dst, bool check_same_device) {
    if (src.GetData() == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input mat is null");
    }

    if (check_same_device && (src.GetDeviceType() != dst.GetDeviceType())) {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }

    if (dst.GetData() == nullptr) {
        dst = Mat(dst.GetDeviceType(), dst.GetMatType(), dst.GetDims());
        if (dst.GetData() == nullptr) {
            return Status(dst.GetData() == nullptr, "dst mat malloc failed.");
        }
    }

    return TNN_OK;
}

static float CalculatePosition(int* position, int i, double scale, int border, int channel) {
    float pos_f = (float)((i + 0.5) * scale - 0.5);
    int pos_i = static_cast<int>(floor(pos_f));
    float rat_f = pos_f - pos_i;
    if (pos_i < 0) {
        pos_i = 0;
        rat_f = 0.f;
    }
    if (pos_i >= border - 1) {
        pos_i = border - 2;
        rat_f = 1.f;
    }
    position[i] = pos_i * channel;

    return rat_f;
}

void CalculatePositionAndRatio(int length, double scale, int border, int channel,
                                         int* position, short* ratio) {
    const int INTER_RESIZE_COEF_BITS  = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    for (int i = 0; i < length; i++) {
        float rat_f = CalculatePosition(position, i, scale, border, channel);
        float a0 = (1.f - rat_f) * INTER_RESIZE_COEF_SCALE;
        float a1 = rat_f * INTER_RESIZE_COEF_SCALE;

        ratio[i * 2]     = SATURATE_CAST_SHORT(a0);
        ratio[i * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }
}

void CalculatePositionAndMask(int length, double scale, int border, int channel,
                                     int* position, uint8_t* mask) {
    for (int i = 0; i < length; i++) {
        float rat_f = CalculatePosition(position, i, scale, border, channel);
        mask[i] = (rat_f <= 0.5) ? -1 : 0;
    }
}

#define  GetResizeBufPreparation(type)                                    \
    double scale_x = (double)src_w / w;                                   \
    double scale_y = (double)src_h / h;                                   \
    *buf = new int[w + h + w + h];                                        \
    int* xofs = *buf;                                                     \
    int* yofs = *buf + w;                                                 \
    type* ialpha = (type*)(*buf + w + h);                                 \
    type* ibeta  = (type*)(*buf + w + h + w);

// Meanings of xofs, yofs, ialpha, ibeta in src image:
//                               |  ialpha[2*x]  |  ialpha[2*x+1]  |
//     --       (xofs[x], yofs[y])                                 (xofs[x]+1, yofs[y])
// ibeta[2*y]
//     --                              (x*scale_x, y*scale_y)
// ibeta[2*y+1]
//     --       (xofs[x], yofs[y]+1)                               (xofs[x]+1, yofs[y]+1)
void GetResizeBuf(int src_w, int src_h, int w, int h, int c, int** buf) {
    GetResizeBufPreparation(short);

    CalculatePositionAndRatio(w, scale_x, src_w, c, xofs, ialpha);
    CalculatePositionAndRatio(h, scale_y, src_h, 1, yofs, ibeta);
}


// Meanings of xofs, yofs, ialpha, ibeta in src image:
//                               |  ialpha[x] (1: left, 0: right)  |
//     --       (xofs[x], yofs[y])                                 (xofs[x]+1, yofs[y])
// ibeta[y]
// (1: top,                            (x*scale_x, y*scale_y)
//  0: bottom)
//     --       (xofs[x], yofs[y]+1)                               (xofs[x]+1, yofs[y]+1)
void GetResizeBufNearset(int src_w, int src_h, int w, int h, int c, int** buf) {
    GetResizeBufPreparation(uint8_t);

    CalculatePositionAndMask(w, scale_x, src_w, c, xofs, ialpha);
    CalculatePositionAndMask(h, scale_y, src_h, 1, yofs, ibeta);
}

inline void InterpolateLinear(float x, float* coeffs) {
    coeffs[0] = 1.f - x;
    coeffs[1] = x;
}

void InitInterTab1D(float* tab, int tabsz) {
    float scale = 1.f / tabsz;
    for (int i = 0; i < tabsz; i++, tab += 2)
        InterpolateLinear(i * scale, tab);
}

void WarpAffineMatrixInverse(const float (*transform)[3], double* inverse) {
    double M[6];
    M[0] = transform[0][0];
    M[1] = transform[0][1];
    M[2] = transform[0][2];
    M[3] = transform[1][0];
    M[4] = transform[1][1];
    M[5] = transform[1][2];

    // Inverse transform matrix
    double D   = M[0] * M[4] - M[1] * M[3];
    D          = D != 0 ? 1. / D : 0;
    double A11 = M[4] * D, A22 = M[0] * D;
    inverse[0]      = A11;
    inverse[1]      = M[1] * (-D);
    inverse[3]      = M[3] * (-D);
    inverse[4]      = A22;
    double b1 = -A11        * M[2] - inverse[1] * M[5];
    double b2 = -inverse[3] * M[2] - A22        * M[5];
    inverse[2]      = b1;
    inverse[5]      = b2;
}

int GetMatElementSize(Mat* mat) {
    MatType mat_type = mat->GetMatType();
    if (NCHW_FLOAT == mat_type) {
        return 4;
    } else if (NC_INT32 == mat_type) {
        return 4;
    } else if (N8UC3 == mat_type || N8UC4 == mat_type || NGRAY == mat_type || NNV21 == mat_type || NNV12 == mat_type) {
        return 1;
    } else if (RESERVED_BFP16_TEST == mat_type || RESERVED_FP16_TEST == mat_type) {
        return 2;
    } else if (RESERVED_INT8_TEST == mat_type) {
        return 1;
    } else {
        return 0;
    }
}

}  // namespace TNN_NS
