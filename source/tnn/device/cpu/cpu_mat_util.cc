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

#include "tnn/device/cpu/cpu_mat_util.h"
#include <algorithm>
#include <type_traits>
#include "tnn/core/macro.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/mat_converter_utils.h"

namespace TNN_NS {

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), 0), UCHAR_MAX)
#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X) (int)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

#define INTER_REMAP_COEF_BITS  15
#define INTER_REMAP_COEF_SCALE (1<<INTER_REMAP_COEF_BITS)
#define INTER_BITS      5
#define INTER_TAB_SIZE  (1<<INTER_BITS)
#define KSIZE 2
static void ResizeGetAdjacentRows(int sy, int prev_sy, short* rows0, short* rows1, int* xofs,
                                     const uint8_t* src, int src_stride, int c, int w, const short* ialphap) {
    const uint8_t* S0 = src + src_stride * (sy);
    const uint8_t* S1 = src + src_stride * (sy + 1);

    short* rows0p        = rows0;
    short* rows1p        = rows1;
    for (int dx = 0; dx < w; dx++) {
        int sx   = xofs[dx];
        short a0 = ialphap[0];
        short a1 = ialphap[1];

        const uint8_t* S0p = S0 + sx;
        const uint8_t* S1p = S1 + sx;

        for (int dc = 0; dc < c; ++dc) {
            rows0p[dc]         = (S0p[dc] * a0 + S0p[dc + c] * a1) >> 4;
            rows1p[dc]         = (S1p[dc] * a0 + S1p[dc + c] * a1) >> 4;
        }

        ialphap += 2;
        rows0p += c;
        rows1p += c;
    }
}

static void ResizeCalculateOneRow(short* rows0p, short* rows1p, const int b0, const int b1, const int w, const int c,
                                     uint8_t* Dp) {
    int remain = w * c;
    for (; remain; --remain) {
        *Dp++ = (uint8_t)(
            ((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
    }
}

void ResizeBilinearImpl(const uint8_t* src, int src_w, int src_h, int src_stride,
                             uint8_t* dst, int w, int h, int stride, int channel) {
    int* buf = nullptr;
    GetResizeBuf(src_w, src_h, w, h, channel, &buf);
    int* xofs = buf;
    int* yofs = buf + w;
    short* ialpha = (short*)(buf + w + h);
    short* ibeta  = (short*)(buf + w + h + w);

    // loop body
    short* rows0 = new short[w * channel];
    short* rows1 = new short[w * channel];

    int prev_sy = -2;

    for (int dy = 0; dy < h; dy++) {
        int sy = yofs[dy];
        ResizeGetAdjacentRows(sy, prev_sy, rows0, rows1, xofs, src, src_stride, channel, w, ialpha);
        prev_sy = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        uint8_t* Dp   = dst + stride * (dy);

        ResizeCalculateOneRow(rows0, rows1, b0, b1, w, channel, Dp);

        ibeta += 2;
    }

    delete[] rows0;
    delete[] rows1;
    delete[] buf;
}

void ResizeNearestImpl(const uint8_t* src, int batch, int src_w, int src_h, int src_stride,
                         uint8_t* dst, int w, int h, int stride, int channel) {
    int* buf      = nullptr;
    GetResizeBufNearset(src_w, src_h, w, h, channel, &buf);
    int* xofs     = buf;
    int* yofs     = buf + w;
    uint8_t* ialpha = (uint8_t*)(buf + w + h);
    uint8_t* ibeta  = (uint8_t*)(buf + w + h + w);

    // loop body
    for (int b = 0; b < batch; ++b) {
        for (int dy = 0; dy < h; dy++) {
            int sy = (ibeta[dy] == 0) ? yofs[dy] + 1 : yofs[dy];

            const uint8_t* Sp = src + src_stride * (b * src_h + sy);
            uint8_t* Dp       = dst + stride * (b * h + dy);

            int dx = 0;
            for (; dx < w; dx++) {
                int sx = xofs[dx];
                for(int dc = 0; dc < channel; dc++) {
                    Dp[dx*channel + dc] = (ialpha[dx] == 0) ? Sp[sx + dc + channel] : Sp[sx + dc];
                }
            }
        }
    }

    delete[] buf;
}

void ResizeBilinear(const uint8_t* src, int src_w, int src_h, uint8_t* dst, int w, int h, int channel) {
    return ResizeBilinearImpl(src, src_w, src_h, src_w * channel, dst, w, h, w * channel, channel);
}

void ResizeNearest(const uint8_t* src, int batch, int src_w, int src_h, uint8_t* dst, int w, int h, int channel) {
    return ResizeNearestImpl(src, batch, src_w, src_h, src_w * channel, dst, w, h, w * channel, channel);
}

bool CheckDataIsOnBoundary(const int new_x_loc, const int new_y_loc, const int src_w, const int src_h) {
    return new_x_loc >= -1 && new_x_loc <= (src_w - 1) &&
           new_y_loc >= -1 && new_y_loc <= (src_h - 1);
}

static void CalculateBilinearOutput(const uint8_t* src, const uint8_t* src2, uint8_t* dst,
                                    int* adelta, int* bdelta, int src_h, int src_w, int channel,
                                    int x, int y, int dst_loc_base, float* _tab, int border_val) {
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
    else if (CheckDataIsOnBoundary(new_x_loc, new_y_loc, src_w, src_h)) {
        int dsc_loc = dst_loc_base + x * channel;

        int mask0 = new_x_loc >= 0 && new_y_loc >= 0;
        int mask1 = new_x_loc <= (src_w - 2) && new_y_loc >= 0;
        int mask2 = new_x_loc >= 0 && new_y_loc <= (src_h - 2);
        int mask3 = new_x_loc <= (src_w - 2) && new_y_loc <= (src_h - 2);

        for (int c = 0; c < channel; ++c) {
            int val_xy = 0;
            val_xy += bilinearWeight[0] * (mask0 ? src[src_loc + c] : border_val);
            val_xy += bilinearWeight[1] * (mask1 ? src[src_loc + channel + c] : border_val);
            val_xy += bilinearWeight[2] * (mask2 ? src2[src_loc + c] : border_val);
            val_xy += bilinearWeight[3] * (mask3 ? src2[src_loc + channel + c] : border_val);
            dst[dsc_loc + c] = SATURATE_CAST_UCHAR((val_xy + (1 << 14)) >> 15);
        }
    }
}

void WarpAffineBilinear(const uint8_t* src, int src_w, int src_h, int channel, uint8_t* dst, int dst_w, int dst_h,
                         const float (*transform)[3], const float border_val)
{
    // Init
    uint8_t border_ival = (uint8_t)border_val;
    for (int i = 0; i < dst_h * dst_w * channel; ++i) {
        dst[i] = border_ival;
    }

    float* _tab = new float[2 * INTER_TAB_SIZE];
    InitInterTab1D(_tab, INTER_TAB_SIZE);

    double m[6];
    WarpAffineMatrixInverse(transform, m);

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

    const uint8_t* src2 = src + src_w * channel;

    for (int y = 0; y < dst_h; ++y) {
        int dst_loc_base    = y * dst_w * channel;

        for (int x = 0; x < dst_w; ++x) {
            CalculateBilinearOutput(src, src2, dst, adelta, bdelta, src_h, src_w, channel, x, y,
                                    dst_loc_base, _tab, (int)border_ival);
        }
    }

    delete[] buf_loc;
    delete[] tab_loc;
    delete[] _tab;

    free(buffer);
}

static void CalculateNearestOutput(const uint8_t* src, const uint8_t* src2, uint8_t* dst,
                                   int* adelta, int* bdelta, int src_h, int src_w, int channel,
                                   int x, int y, int dst_loc_base, int border_val) {
    int new_x       = adelta[2 * x] + bdelta[2 * y] + 16;
    int new_y       = adelta[2 * x + 1] + bdelta[2 * y + 1] + 16;
    int new_x_loc   = new_x >> 10;
    int new_y_loc   = new_y >> 10;

    short coeffs_x  = (new_x >> 5) & 31;
    short coeffs_y  = (new_y >> 5) & 31;

    int src_loc     = (new_x_loc + new_y_loc * src_w) * channel;

    if (new_x_loc >= 0 && new_x_loc < (src_w - 1) && new_y_loc >= 0 && new_y_loc < (src_h - 1)) {
        for (int c = 0; c < channel; c++)
        {
            int dst_loc = dst_loc_base + x * channel;
            int point00 = src[src_loc + c];
            int point01 = src[src_loc + channel + c];
            int point10 = src2[src_loc + c];
            int point11 = src2[src_loc + channel + c];

            int val_xy;
            if (coeffs_y < (1<<4)) {
                val_xy = (coeffs_x < (1<<4)) ? point00 : point01;
            } else {
                val_xy = (coeffs_x < (1<<4)) ? point10 : point11;
            }

            dst[dst_loc + c] = val_xy;
        }
    }
    else if (CheckDataIsOnBoundary(new_x_loc, new_y_loc, src_w, src_h)) {
        int dsc_loc = dst_loc_base + x * channel;

        int mask0 = new_x_loc >= 0 && new_y_loc >= 0;
        int mask1 = new_x_loc <= (src_w - 2) && new_y_loc >= 0;
        int mask2 = new_x_loc >= 0 && new_y_loc <= (src_h - 2);
        int mask3 = new_x_loc <= (src_w - 2) && new_y_loc <= (src_h - 2);

        for (int c = 0; c < channel; ++c) {
            int point00 = mask0 ? src[src_loc + c] : border_val;
            int point01 = mask1 ? src[src_loc + channel + c] : border_val;
            int point10 = mask2 ? src2[src_loc + c] : border_val;
            int point11 = mask3 ? src2[src_loc + channel + c] : border_val;

            int val_xy = 0;
            if (coeffs_y < (1<<4)) {
                val_xy = (coeffs_x < (1<<4)) ? point00 : point01;
            } else {
                val_xy = (coeffs_x < (1<<4)) ? point10 : point11;
            }

            dst[dsc_loc + c] = val_xy;
        }
    }
}

void WarpAffineNearest(const uint8_t* src, int src_w, int src_h, int channel, uint8_t* dst, int dst_w, int dst_h,
                       const float (*transform)[3], const float border_val)
{
    // Init
    uint8_t border_ival = (uint8_t)border_val;
    for (int i = 0; i < dst_h * dst_w * channel; ++i) {
        dst[i] = border_ival;
    }

    double m[6];
    WarpAffineMatrixInverse(transform, m);

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

    const uint8_t* src2 = src + src_w * channel;

    for (int y = 0; y < dst_h; ++y) {
        int dst_loc_base    = y * dst_w * channel;

        for (int x = 0; x < dst_w; ++x) {
            CalculateNearestOutput(src, src2, dst, adelta, bdelta, src_h, src_w, channel, x, y,
                                   dst_loc_base, (int)border_ival);
        }
    }

    free(buffer);
}

void BGROrBGRAToGray(const uint8_t* src, uint8_t* dst, int h, int w, int channel) {
    NaiveBGROrBGRAToGray(src, dst, h, w, channel);
}

#undef SATURATE_CAST_UCHAR

void YUVToBGR(const unsigned char* yuv, unsigned char* bgr, int h, int w, bool is_nv12) {
    NaiveYUVToBGROrBGRA(yuv, bgr, 3, h, w, is_nv12);
}

void YUVToBGRA(const unsigned char* yuv, unsigned char* bgra, int h, int w, bool is_nv12) {
    NaiveYUVToBGROrBGRA(yuv, bgra, 4, h, w, is_nv12);
}

}  // namespace TNN_NS
