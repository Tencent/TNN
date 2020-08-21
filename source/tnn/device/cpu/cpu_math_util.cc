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

#include "tnn/device/cpu/cpu_math_util.h"

#include <type_traits>

#include "tnn/core/macro.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), 0), UCHAR_MAX)
#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X) (int)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

#define INTER_REMAP_COEF_BITS  15
#define INTER_REMAP_COEF_SCALE (1<<INTER_REMAP_COEF_BITS)
#define INTER_BITS      5
#define INTER_TAB_SIZE  (1<<INTER_BITS)
#define KSIZE 2

static inline void interpolateLinear(float x, float* coeffs) {
    coeffs[0] = 1.f - x;
    coeffs[1] = x;
}

static void initInterTab1D(float* tab, int tabsz) {
    float scale = 1.f / tabsz;
    for (int i = 0; i < tabsz; i++, tab += 2)
        interpolateLinear(i * scale, tab);
}

void warpaffine_bilinear(const uint8_t* src, int src_w, int src_h, int channel, uint8_t* dst, int dst_w, int dst_h,
                         const float (*transform)[3], const float border_val)
{
    // Init
    uint8_t border_ival = (uint8_t)border_val;
    for (int i = 0; i < dst_h * dst_w * channel; ++i) {
        dst[i] = border_ival;
    }

    float* _tab = new float[2 * INTER_TAB_SIZE];
    initInterTab1D(_tab, INTER_TAB_SIZE);

    double m[6];
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
    m[0]      = A11;
    m[1]      = M[1] * (-D);
    m[3]      = M[3] * (-D);
    m[4]      = A22;
    double b1 = -A11 * M[2] - m[1] * M[5];
    double b2 = -m[3] * M[2] - A22 * M[5];
    m[2]      = b1;
    m[5]      = b2;

    int* buffer = (int *)malloc((dst_w + dst_h) * 2 * sizeof(int));

    int* adelta = buffer;
    int* bdelta = buffer + dst_w * 2;

    for (int x = 0; x < dst_w; x++) {
        adelta[x * 2] = SATURATE_CAST_INT(m[0] * x * 1024);
        adelta[x * 2 + 1] = SATURATE_CAST_INT(m[3] * x * 1024);
    }

    for (int y = 0; y < dst_h; y++) {
        bdelta[y * 2] = SATURATE_CAST_INT((m[1] * y + m[2]) * 1024);
        bdelta[y * 2 + 1] = SATURATE_CAST_INT((m[4] * y + m[5]) * 1024);
    }

    int* buf_loc   = new int[dst_w];
    short* tab_loc = new short[dst_w];

    const unsigned char* src2 = src + src_w * channel;

    for (int y = 0; y < dst_h; ++y) {
        int dst_loc_base    = y * dst_w * channel;

        for (int x = 0; x < dst_w; ++x) {
            int new_x       = adelta[2 * x] + bdelta[2 * y] + 16;
            int new_y       = adelta[2 * x + 1] + bdelta[2 * y + 1] + 16;
            int new_x_loc   = new_x >> 10;
            int new_y_loc   = new_y >> 10;

            short coeffs_x  = (new_x >> 5) & 31;
            short coeffs_y  = (new_y >> 5) & 31;

            int src_loc     = (new_x_loc + new_y_loc * src_w) * channel;

            short bilinearWeight[KSIZE * KSIZE];
            // set weight for bilinear
            for (int yy = 0; yy < KSIZE; yy++)
            {
                float vy    = _tab[coeffs_y * KSIZE + yy];
                for (int xx = 0; xx < KSIZE; xx++)
                {
                    float v = vy * _tab[coeffs_x * KSIZE + xx];
                    bilinearWeight[yy * KSIZE + xx] = SATURATE_CAST_SHORT(v * INTER_REMAP_COEF_SCALE);
                }
            }

            if (new_x_loc >= 0 && new_x_loc < (src_w - 1) && new_y_loc >= 0 && new_y_loc < (src_h - 1)) {
                for (int c = 0; c < channel; c++)
                {
                    int dst_loc = dst_loc_base + x * channel;
                    int point00 = src[src_loc + c];
                    int point01 = src[src_loc + channel + c];
                    int point10 = src2[src_loc + c];
                    int point11 = src2[src_loc + channel + c];

                    int val_xy  = bilinearWeight[0] * point00 + bilinearWeight[1] * point01 + bilinearWeight[2] * point10 +
                                  bilinearWeight[3] * point11;

                    dst[dst_loc + c] = SATURATE_CAST_UCHAR((val_xy + (1 << 14)) >> 15);
                }
            }
            else if (new_x_loc >= -1 && new_x_loc <= (src_w - 1) &&
                     new_y_loc >= -1 && new_y_loc <= (src_h - 1)) {
                int dsc_loc = dst_loc_base + x * channel;

                int mask0 = new_x_loc >= 0 && new_y_loc >= 0;
                int mask1 = new_x_loc <= (src_w - 2) && new_y_loc >= 0;
                int mask2 = new_x_loc >= 0 && new_y_loc <= (src_h - 2);
                int mask3 = new_x_loc <= (src_w - 2) && new_y_loc <= (src_h - 2);

                for (int c = 0; c < channel; ++c) {
                    int val_xy = 0;
                    if (mask0) {
                        val_xy += bilinearWeight[0] * src[src_loc + c];
                    }
                    if (mask1) {
                        val_xy += bilinearWeight[1] * src[src_loc + channel + c];
                    }
                    if (mask2) {
                        val_xy += bilinearWeight[2] * src2[src_loc + c];
                    }
                    if (mask3) {
                        val_xy += bilinearWeight[3] * src2[src_loc + channel + c];
                    }
                    dst[dsc_loc + c] = SATURATE_CAST_UCHAR((val_xy + (1 << 14)) >> 15);
                }
            }
        }
    }

    delete[] buf_loc;
    delete[] tab_loc;
    delete[] _tab;

    free(buffer);
}

}  // namespace TNN_NS
