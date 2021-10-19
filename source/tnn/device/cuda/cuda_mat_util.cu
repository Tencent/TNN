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

#include "tnn/device/cuda/cuda_mat_util.cuh"

#include <algorithm>

#include "tnn/utils/mat_utils.h"
#include "tnn/utils/mat_converter_utils.h"
#include "tnn/device/cuda/cuda_macro.h"
#include "tnn/device/cuda/cuda_mat_util.cuh"
#include "cuda_runtime.h"

namespace TNN_NS {

#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX)
#define INTER_REMAP_COEF_BITS  15
#define INTER_REMAP_COEF_SCALE (1<<INTER_REMAP_COEF_BITS)
#define INTER_BITS      5
#define INTER_TAB_SIZE  (1<<INTER_BITS)
#define KSIZE 2

template<int ELE_PER_THREAD>
__global__ void resize_bilinear_kernel(uint8_t* dst_data, int dst_hwc, int height, int width, const uint8_t* src_data,
        int src_height, int src_width, int src_hwc, int channel, float scale_x, float scale_y) {
    int ele_off = ELE_PER_THREAD * blockDim.x * blockIdx.x + threadIdx.x;

    src_data += blockIdx.y * src_hwc;
    dst_data += blockIdx.y * dst_hwc + ele_off;
    int index = ele_off;

    #pragma unroll
    for (int i = 0; i < ELE_PER_THREAD; i++) {
        if (index < dst_hwc) {
            const int h = index / channel / width;
            const int w = (index / channel) % width;
            const int c = index % channel;

            float fy = (float)((h + 0.5) * scale_y - 0.5);
            int sy = __float2int_rd(fy);
            fy -= sy;
            int sy2 = sy + 1;
            if (sy < 0) {
                sy = sy2 = 0;
                fy = 0;
            }
            if (sy >= src_height - 1) {
                sy = sy2 = src_height - 1;
                fy = 0;
            }
            int cbufy_0 = __float2int_rn((1.f - fy) * INTER_RESIZE_COEF_SCALE);
            int cbufy_1 = __float2int_rn(fy * INTER_RESIZE_COEF_SCALE);

            float fx = (float)((w + 0.5) * scale_x - 0.5);
            int sx = __float2int_rd(fx);
            fx -= sx;
            int sx2 = sx + 1;
            if (sx < 0) {
                fx = 0;
                sx = sx2 = 0;
            }
            if (sx >= src_width - 1) {
                fx = 0;
                sx = sx2 = src_width - 1;
            }
            int cbufx_0 = __float2int_rn((1.f - fx) * INTER_RESIZE_COEF_SCALE);
            int cbufx_1 = __float2int_rn(fx * INTER_RESIZE_COEF_SCALE);
            int src_idx0 = sy * src_width * channel + sx * channel + c;
            int src_idx1 = sy * src_width * channel + sx2 * channel + c;
            int src_idx2 = sy2 * src_width * channel + sx * channel + c;
            int src_idx3 = sy2 * src_width * channel + sx2 * channel + c;
            int s0 = (src_data[src_idx0] * cbufx_0 + src_data[src_idx1] * cbufx_1) >> 4;
            int s1 = (src_data[src_idx2] * cbufx_0 + src_data[src_idx3] * cbufx_1) >> 4;
            uint8_t val = ((s0 * cbufy_0 >> 16) + (s1 * cbufy_1 >> 16) + 2) >> 2;
            dst_data[i * blockDim.x] = val;
        }
        index += blockDim.x;
    }
}

template<int ELE_PER_THREAD>
__global__ void resize_nearest_kernel(uint8_t* dst_data, int dst_hwc, int height, int width, const uint8_t* src_data,
        int src_height, int src_width, int src_hwc, int channel, float scale_x, float scale_y) {
    int ele_off = ELE_PER_THREAD * blockDim.x * blockIdx.x + threadIdx.x;

    src_data += blockIdx.y * src_hwc;
    dst_data += blockIdx.y * dst_hwc + ele_off;
    int index = ele_off;

    #pragma unroll
    for (int i = 0; i < ELE_PER_THREAD; i++) {
        if (index < dst_hwc) {
            const int h = index / channel / width;
            const int w = (index / channel) % width;
            const int c = index % channel;

            float pos_fx = (float)((w + 0.5) * scale_x - 0.5);
            int pos_ix = (int)pos_fx;
            float rat_fx = pos_fx - pos_ix;
            if (pos_ix < 0) {
                pos_ix = 0;
                rat_fx = 0.f;
            }
            if (pos_ix >= src_width - 1) {
                pos_ix = src_width - 2;
                rat_fx = 1.f;
            }
            int mask_x = (rat_fx <= 0.5) ? -1 : 0;

            float pos_fy = (float)((h + 0.5) * scale_y - 0.5);
            int pos_iy = (int)pos_fy;
            float rat_fy = pos_fy - pos_iy;
            if (pos_iy < 0) {
                pos_iy = 0;
                rat_fy = 0.f;
            }
            if (pos_iy >= src_height - 1) {
                pos_iy = src_height - 2;
                rat_fy = 1.f;
            }
            int mask_y = (rat_fy <= 0.5) ? -1 : 0;

            int sy = (mask_y == 0) ? pos_iy + 1 : pos_iy;
            int sx = pos_ix;
            int src_idx = sy * src_width * channel + sx * channel + c;
            dst_data[i * blockDim.x] = (mask_x == 0) ? src_data[src_idx + channel] : src_data[src_idx];
        }
        index += blockDim.x;
    }
}

__global__ void crop_rgb_kernel(const uint8_t* src, uint8_t* dst, int channel, int src_width, int src_height, int dst_width,
        int dst_height, int width, int height, int top_left_x, int top_left_y) {
    int batch = blockIdx.y;
    int ele_off = blockDim.x * blockIdx.x + threadIdx.x;

    if (ele_off < channel * width * height) {
        src += batch * src_width * src_height * channel + (top_left_x + top_left_y * src_width) * channel;
        dst += batch * dst_width * dst_height * channel;
        int h = ele_off / (width * channel);
        int w = ele_off % (width * channel);
        dst[dst_width * h * channel + w] = src[src_width * h * channel + w];
    }
}

__global__ void crop_yuv_kernel(const uint8_t* src, uint8_t* dst, int src_width, int src_height, int dst_width,
        int dst_height, int width, int height, int top_left_x, int top_left_y) {
    int batch = blockIdx.y;
    int ele_off = blockDim.x * blockIdx.x + threadIdx.x;

    if (ele_off < width * height) {
        const uint8_t* src_y = src + batch * src_height * src_width * 3 / 2 + top_left_x + top_left_y * src_width;
        uint8_t* dst_y = dst + batch * dst_height * dst_width * 3 / 2;
        int h = ele_off / width;
        int w = ele_off % width;
        dst_y[dst_width * h + w] = src_y[src_width * h + w];
        if (h < height / 2) {
            const uint8_t* src_uv = src + batch * src_height * src_width * 3 / 2 + src_width * src_height +
                top_left_x + top_left_y * src_width / 2;
            uint8_t* dst_uv = dst + batch * dst_height * dst_width * 3 / 2 + dst_width * dst_height;
            dst_uv[dst_width * h + w] = src_uv[src_width * h + w];
        }
    }
}

__global__ void yuv_to_rgba_kernel(const uint8_t* src, uint8_t* dst, int height, int width, int HW, int channel, bool is_nv12) {
    src += blockIdx.z * HW * 3 / 2;
    dst += blockIdx.z * HW * channel;

    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (w < width && h < height) {
        uint8_t y = src[h * width + w];
        uint8_t u, v;
        if (is_nv12) {
            u = src[HW + (h / 2 * (width / 2 * 2)) + w / 2 * 2];
            v = src[HW + (h / 2 * (width / 2 * 2)) + w / 2 * 2 + 1];
        } else {
            v = src[HW + (h / 2 * (width / 2 * 2)) + w / 2 * 2];
            u = src[HW + (h / 2 * (width / 2 * 2)) + w / 2 * 2 + 1];
        }

        int b = 1.164 * (y - 16) + 2.018 * (u - 128);
        int g = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128);
        int r = 1.164 * (y - 16) + 1.596 * (v - 128);
        if (r > 255)   r = 255;
        if (g > 255)   g = 255;
        if (b > 255)   b = 255;
        if (r < 0)     r = 0;
        if (g < 0)     g = 0;
        if (b < 0)     b = 0;

        uint8_t* rgb = dst + (h * width + w) * channel;
        rgb[0] = (uint8_t)r;
        rgb[1] = (uint8_t)g;
        rgb[2] = (uint8_t)b;
        if (channel == 4)
            rgb[3] = 255;
    }
}

__global__ void bgra_to_gray_kernel(const uint8_t* src, uint8_t* dst, int height, int width, int HW, int channel) {
    src += blockIdx.z * HW * channel;
    dst += blockIdx.z * HW;

    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (w < width && h < height) {
        uint8_t b = src[(h * width + w) * channel + 0];
        uint8_t g = src[(h * width + w) * channel + 1];
        uint8_t r = src[(h * width + w) * channel + 2];
        float gray_color = 0.114f * b + 0.587 * g + 0.299 * r;
        dst[h * width + w] = gray_color;
    }
}

__global__ void copy_make_border_kernel(const uint8_t* src, uint8_t* dst, int src_height, int src_width, int dst_height,
        int dst_width, int top, int bottom, int left, int right, uint8_t pad_val) {
    src += blockIdx.y * src_height * src_width;
    dst += blockIdx.y * dst_height * dst_width;
    int offset = blockDim.x * blockIdx.x + threadIdx.x;

    if (offset < src_height * src_width) {
        int h = offset / src_width;
        int w = offset % src_width;
        if (h == 0) {
            if (w == 0) {
                for (int i = 0; i < top; i++) {
                    for (int j = 0; j < left; j++) {
                        dst[i * dst_width + j] = pad_val;
                    }
                }
            }
            for (int i = 0; i < top; i++) {
                dst[left + i * dst_width + w] = pad_val;
            }
            if (w == src_width - 1) {
                for (int i = 0; i < top; i++) {
                    for (int j = 0; j < right; j++) {
                        dst[i * dst_width + left + src_width + j] = pad_val;
                    }
                }
            }
        }
        if (w == 0) {
            for (int i = 0; i < left; i++) {
                dst[(top + h) * dst_width + i] = pad_val;
            }
        }
        if (w == src_width - 1) {
            for (int i = 0; i < right; i++) {
                dst[(top + h) * dst_width + left + src_width + i] = pad_val;
            }
        }
        if (h == src_height - 1) {
            if (w == 0) {
                for (int i = 0; i < bottom; i++) {
                    for (int j = 0; j < left; j++) {
                        dst[(top + src_height + i) * dst_width + j] = pad_val;
                    }
                }
            }
            for (int i = 0; i < bottom; i++) {
                dst[left + (top + src_height + i) * dst_width + w] = pad_val;
            }
            if (w == src_width - 1) {
                for (int i = 0; i < bottom; i++) {
                    for (int j = 0; j < right; j++) {
                        dst[(top + src_height + i) * dst_width + left + src_width + j] = pad_val;
                    }
                }
            }
        }
        dst[(h + top) * dst_width + left + w] = src[h * src_width + w];
    }
}

__device__ int fp_2_int_sat(double in) {
    long long x = __double2ll_rn(in);
    x = x > INT_MAX?INT_MAX : x;
    x = x < INT_MIN?INT_MIN : x;
    return int(x);
}

__device__ __forceinline__ int imax(int a, int b) {
    return max(a,b);
}

__device__ __forceinline__ int imin(int a, int b) {
    return min(a,b);
}

template<int ELE_PER_THREAD, int THREAD_PER_BLOCK, BorderType border_type>
__global__ void warp_affine_bilinear_kernel(const uint8_t* src, uint8_t* dst, const int H, const int W, const int C,
        const int OH, const int OW, const short* table, double* tm, const uint8_t border_value) {
    src += blockIdx.y * H * W * C;
    dst += blockIdx.y * OH * OW * C;
    const int DELTA = 1 << 14;

    #pragma unroll
    for (int i = 0; i < ELE_PER_THREAD; i++) {
        const int hwc_id = blockIdx.x * THREAD_PER_BLOCK * ELE_PER_THREAD + 
                            threadIdx.x + i * THREAD_PER_BLOCK;
        if (hwc_id >= OH * OW * C) {
            break;
        }

        // output hwc
        const int c = hwc_id % C;
        const int hw = hwc_id / C;
        const int h = hw / OW;
        const int w = hw % OW;

        int new_w_full = fp_2_int_sat(tm[0] * w * 1024) +
                         fp_2_int_sat((tm[1] * h + tm[2])* 1024) + 16;
        int new_h_full = fp_2_int_sat(tm[3] * w * 1024) +
                         fp_2_int_sat((tm[4] * h + tm[5])* 1024) + 16;

        new_w_full >>= 5;
        new_h_full >>= 5;
        short new_hw_float = (new_w_full & 31) + (new_h_full & 31) * 32;
        const short *wtab = &table[new_hw_float*4];

        int new_w_int = new_w_full >> 5;
        int new_h_int = new_h_full >> 5;

        // input hw
        int new_w_real[2] = { new_w_int, new_w_int + 1 };
        int new_h_real[2] = { new_h_int, new_h_int + 1 };

        unsigned char val[2][2];
        #pragma unroll
        for(int wi = 0; wi < 2; wi++) {
            #pragma unroll
            for(int hi = 0; hi < 2; hi++) {

                if (new_w_real[wi] >= 0 && new_w_real[wi] < W &&
                    new_h_real[hi] >= 0 && new_h_real[hi] < H) {
                    val[hi][wi] = src[(new_h_real[hi] * W + new_w_real[wi]) * C + c];
                } else {
                    switch (border_type) {
                        case BORDER_TYPE_CONSTANT:
                            val[hi][wi] = border_value;
                            break;
                        case BORDER_TYPE_TRANSPARENT:
                            val[hi][wi] = dst[hwc_id];
                            break;
                        default:
                            val[hi][wi] = 0;
                            break;
                    }   
                }
            }
        }

        int val_inter = wtab[0] * val[0][0] + wtab[1] * val[0][1] + wtab[2] * val[1][0] + wtab[3] * val[1][1];
        int src_value = (val_inter + DELTA ) >> 15;
        dst[hwc_id] = src_value;
    }
}

template<int ELE_PER_THREAD, int THREAD_PER_BLOCK, BorderType border_type>
__global__ void warp_affine_nearest_kernel(const uint8_t* src, uint8_t* dst, const int H, const int W, const int C,
        const int OH, const int OW, double* tm, const uint8_t border_value) {
    src += blockIdx.y * H * W * C;
    dst += blockIdx.y * OH * OW * C;

    #pragma unroll
    for (int i = 0; i < ELE_PER_THREAD; i++) {
        const int hwc_id = blockIdx.x * THREAD_PER_BLOCK * ELE_PER_THREAD +
                            threadIdx.x + i * THREAD_PER_BLOCK;
        if (hwc_id >= OH * OW * C) {
            break;
        }

        // output hwc
        const int c = hwc_id % C;
        const int hw = hwc_id / C;
        const int h = hw / OW;
        const int w = hw % OW;

        int new_w_full = fp_2_int_sat(tm[0] * w * 1024) +
                         fp_2_int_sat((tm[1] * h + tm[2])* 1024) + 16;
        int new_h_full = fp_2_int_sat(tm[3] * w * 1024) +
                         fp_2_int_sat((tm[4] * h + tm[5])* 1024) + 16;

        new_w_full >>= 5;
        new_h_full >>= 5;
        bool is_left = (new_w_full & 31) < 16;
        bool is_top = (new_h_full & 31) < 16;
        int new_w_int = new_w_full >> 5;
        int new_h_int = new_h_full >> 5;

        // input hw
        int new_w_real[2] = { new_w_int, new_w_int + 1 };
        int new_h_real[2] = { new_h_int, new_h_int + 1 };

        unsigned char  val[2][2];
        #pragma unroll 
        for(int wi = 0; wi < 2; wi++) {
            #pragma unroll 
            for(int hi = 0; hi < 2; hi++) {
                if (new_w_real[wi] >= 0 && new_w_real[wi] < W && 
                    new_h_real[hi] >= 0 && new_h_real[hi] < H) {
                    val[hi][wi] = src[(new_h_real[hi] * W + new_w_real[wi]) * C + c];
                } else {
                    switch (border_type) {
                        case BORDER_TYPE_CONSTANT:
                            val[hi][wi] = border_value;
                            break;
                        case BORDER_TYPE_TRANSPARENT:
                            val[hi][wi] = dst[hwc_id];
                            break;
                        default:
                            val[hi][wi] = 0;
                            break;
                    }
                }
            }
        }

        if (is_top) {
            dst[hwc_id] = is_left ? val[0][0] : val[0][1];
        } else {
            dst[hwc_id] = is_left ? val[1][0] : val[1][1];
        }
    }
}

void ResizeBilinear(const uint8_t* src, uint8_t* dst, int batch, int src_w, int src_h, int dst_w, int dst_h, int channel) {
    const int ELE_PER_THREAD = 4;
    const int THREAD_PER_BLOCK = 128;
    dim3 grid;
    int size_dst = dst_h * dst_w * channel;
    grid.x = (size_dst + ELE_PER_THREAD * THREAD_PER_BLOCK - 1) / (ELE_PER_THREAD * THREAD_PER_BLOCK);
    grid.y = batch;
    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;
    resize_bilinear_kernel<ELE_PER_THREAD><<<grid, THREAD_PER_BLOCK>>>(dst, size_dst, dst_h, dst_w, src, src_h,
        src_w, src_h * src_w * channel, channel, scale_x, scale_y);
}

void ResizeNearest(const uint8_t* src, uint8_t* dst, int batch, int src_w, int src_h, int dst_w, int dst_h, int channel) {
    const int ELE_PER_THREAD = 4;
    const int THREAD_PER_BLOCK = 128;
    dim3 grid;
    int size_dst = dst_h * dst_w * channel;
    grid.x = (size_dst + ELE_PER_THREAD * THREAD_PER_BLOCK - 1) / (ELE_PER_THREAD * THREAD_PER_BLOCK);
    grid.y = batch;
    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;
    resize_nearest_kernel<ELE_PER_THREAD><<<grid, THREAD_PER_BLOCK>>>(dst, size_dst, dst_h, dst_w, src, src_h,
        src_w, src_h * src_w * channel, channel, scale_x, scale_y);
}

static void initInterTab2D(short* input_table) {
    short* itab = input_table;
    int ksize = 2;

    float *_tab = new float[2 * INTER_TAB_SIZE];
    int i, j, k1, k2;
    InitInterTab1D(_tab, INTER_TAB_SIZE);
    for (i = 0; i < INTER_TAB_SIZE; i++) {
        for (j = 0; j < INTER_TAB_SIZE; j++, itab += ksize * ksize) {
            int isum = 0;

            for (k1 = 0; k1 < ksize; k1++) {
                float vy = _tab[i * ksize + k1];
                for (k2 = 0; k2 < ksize; k2++) {
                    float v = vy * _tab[j * ksize + k2];
                    isum += itab[k1 * ksize + k2] = SATURATE_CAST_SHORT(v * INTER_REMAP_COEF_SCALE);
                }
            }

            if (isum != INTER_REMAP_COEF_SCALE) {
                int diff = isum - INTER_REMAP_COEF_SCALE;
                int ksize2 = ksize / 2;
                int Mk1 = ksize2, Mk2 = ksize2, mk1 = ksize2, mk2 = ksize2;
                for (k1 = ksize2; k1 < ksize2 + 2; k1++)
                    for (k2 = ksize2; k2 < ksize2 + 2; k2++) {
                        if (itab[k1 * ksize + k2] < itab[mk1 * ksize + mk2])
                            mk1 = k1, mk2 = k2;
                        else if (itab[k1 * ksize + k2] > itab[Mk1 * ksize + Mk2])
                            Mk1 = k1, Mk2 = k2;
                    }
                if (diff < 0)
                    itab[Mk1 * ksize + Mk2] = (short)(itab[Mk1 * ksize + Mk2] - diff);
                else
                    itab[mk1 * ksize + mk2] = (short)(itab[mk1 * ksize + mk2] - diff);
            }
        }
    }
    delete[] _tab;
}

void WarpAffineBilinear(const uint8_t* src, int batch, int channel, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
        const float (*transform)[3], const float border_val, BorderType border_type, void* stream) {
    double m[6];
    WarpAffineMatrixInverse(transform, m);
    double *tm_gpu;
    CUDA_CHECK(cudaMalloc((void**)&tm_gpu, 6 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(tm_gpu, m, 6 * sizeof(double), cudaMemcpyHostToDevice));
    const int table_size = INTER_TAB_SIZE * INTER_TAB_SIZE * KSIZE * KSIZE;
    short table_cpu[table_size];
    short *table_gpu;
    CUDA_CHECK(cudaMalloc((void**)&table_gpu, table_size * sizeof(short)));
    initInterTab2D(table_cpu);
    cudaMemcpy(table_gpu, table_cpu, table_size * sizeof(short), cudaMemcpyHostToDevice);
    const int THREAD_PER_BLOCK = 128;
    const int ELE_PER_THREAD = 32;
    int size_dst = dst_h * dst_w * channel;
    dim3 griddim;
    griddim.x = (size_dst + ELE_PER_THREAD * THREAD_PER_BLOCK - 1) / (ELE_PER_THREAD * THREAD_PER_BLOCK);
    griddim.y = batch;
    if (border_type == BORDER_TYPE_CONSTANT) {
        warp_affine_bilinear_kernel<ELE_PER_THREAD, THREAD_PER_BLOCK, BORDER_TYPE_CONSTANT><<<griddim, THREAD_PER_BLOCK, 0, (CUstream_st*)stream>>>(src, dst, src_h, src_w,
            channel, dst_h, dst_w, table_gpu, tm_gpu, border_val);
    } else if (border_type == BORDER_TYPE_TRANSPARENT) {
        warp_affine_bilinear_kernel<ELE_PER_THREAD, THREAD_PER_BLOCK, BORDER_TYPE_TRANSPARENT><<<griddim, THREAD_PER_BLOCK, 0, (CUstream_st*)stream>>>(src, dst, src_h, src_w,
            channel, dst_h, dst_w, table_gpu, tm_gpu, border_val);
    }
    CUDA_CHECK(cudaFree(tm_gpu));
    CUDA_CHECK(cudaFree(table_gpu));
}

void WarpAffineNearest(const uint8_t* src, int batch, int channel, int src_w, int src_h, uint8_t* dst, int dst_w, int dst_h,
        const float (*transform)[3], const float border_val, BorderType border_type) {
    double m[6];
    WarpAffineMatrixInverse(transform, m);
    double *tm_gpu;
    CUDA_CHECK(cudaMalloc((void**)&tm_gpu, 6 * sizeof(double)));
    cudaMemcpy(tm_gpu, m, 6 * sizeof(double), cudaMemcpyHostToDevice);
    const int THREAD_PER_BLOCK = 128;
    const int ELE_PER_THREAD = 8;
    int size_dst = dst_h * dst_w * channel;
    dim3 griddim;
    griddim.x = (size_dst + ELE_PER_THREAD * THREAD_PER_BLOCK - 1) / (ELE_PER_THREAD * THREAD_PER_BLOCK);
    griddim.y = batch;
    if (border_type == BORDER_TYPE_CONSTANT) {
        warp_affine_nearest_kernel<ELE_PER_THREAD, THREAD_PER_BLOCK, BORDER_TYPE_CONSTANT><<<griddim, THREAD_PER_BLOCK>>>(src, dst, src_h, src_w,
            channel, dst_h, dst_w, tm_gpu, border_val);
    } else if (border_type == BORDER_TYPE_TRANSPARENT) {
        warp_affine_nearest_kernel<ELE_PER_THREAD, THREAD_PER_BLOCK, BORDER_TYPE_TRANSPARENT><<<griddim, THREAD_PER_BLOCK>>>(src, dst, src_h, src_w,
            channel, dst_h, dst_w, tm_gpu, border_val);
    }
    CUDA_CHECK(cudaFree(tm_gpu));
}

void CropRGB(const uint8_t* src, uint8_t* dst, int batch, int channel, int src_width, int src_height, int dst_width, int dst_height,
        int width, int height, int top_left_x, int top_left_y) {
    int THREAD_PER_BLOCK = 128;
    dim3 grid;
    grid.x = (width * height * channel + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    grid.y = batch;
    crop_rgb_kernel<<<grid, THREAD_PER_BLOCK>>>(src, dst, channel, src_width, src_height, dst_width, dst_height,
        width, height, top_left_x, top_left_y);
}

void CropYUV(const uint8_t* src, uint8_t* dst, int batch, int src_width, int src_height, int dst_width, int dst_height,
        int width, int height, int top_left_x, int top_left_y) {
    int THREAD_PER_BLOCK = 128;
    dim3 grid;
    grid.x = (width * height + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    grid.y = batch;
    crop_yuv_kernel<<<grid, THREAD_PER_BLOCK>>>(src, dst, src_width, src_height, dst_width, dst_height, width,
        height, top_left_x, top_left_y);
}

void YUVToGRBA(const uint8_t* src, uint8_t* dst, int batch, int h, int w, int channel, bool is_nv12) {
    dim3 block, grid;
    int BLOCKX = 32;
    int BLOCKY = 8;
    block.x = BLOCKX;
    block.y = BLOCKY;
    grid.x = (w + BLOCKX - 1) / BLOCKX;
    grid.y = (h + BLOCKY - 1) / BLOCKY;
    grid.z = batch;
    yuv_to_rgba_kernel<<<grid, block>>>(src, dst, h, w, w * h, channel, is_nv12);
}

void BGRAToGRAY(const uint8_t* src, uint8_t* dst, int batch, int h, int w, int channel) {
    dim3 block, grid;
    int BLOCKX = 32;
    int BLOCKY = 8;
    block.x = BLOCKX;
    block.y = BLOCKY;
    grid.x = (w + BLOCKX - 1) / BLOCKX;
    grid.y = (h + BLOCKY - 1) / BLOCKY;
    grid.z = batch;
    bgra_to_gray_kernel<<<grid, block>>>(src, dst, h, w, w * h, channel);
}

void CudaCopyMakeBorder(const uint8_t* src, uint8_t* dst, int batch, int src_width, int src_height, int dst_width,
        int dst_height, int channel, int top, int bottom, int left, int right, uint8_t pad_val) {
    int THREAD_PER_BLOCK = 128;
    dim3 grid;
    grid.x = (src_width * src_height * channel + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    grid.y = batch;
    copy_make_border_kernel<<<grid, THREAD_PER_BLOCK>>>(src, dst, src_height, src_width * channel, dst_height, dst_width * channel, top, bottom,
        left * channel, right * channel, pad_val);
}

}  //  namespace TNN_NS

