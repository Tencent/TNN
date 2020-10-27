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

#include "tnn/device/arm/acc/compute/compute.h"

#include <string.h>

#include "tnn/core/macro.h"
#include "tnn/device/arm/acc/Float4.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_util.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

/*
add bias
*/
template <typename T>
void PostAddBias(void* dst, const float* bias, long area, long oc4) {
    for (long z = oc4 - 1; z >= 0; --z) {
        Float4 vbias = Float4::load(bias + 4 * z);
        auto dst_z   = reinterpret_cast<T*>(dst) + area * 4 * z;
        long p       = 0;
        for (; p < area - 3; p += 4) {
            auto dst_p = dst_z + 4 * p;
            Float4x4::save(dst_p, Float4x4::load(dst_p) + vbias);
        }
        for (; p < area; ++p) {
            auto dst_p = dst_z + 4 * p;
            Float4::save(dst_p, Float4::load(dst_p) + vbias);
        }
    }
}
template void PostAddBias<float>(void* dst, const float* bias, long area, long oc4);
template void PostAddBias<bfp16_t>(void* dst, const float* bias, long area, long oc4);

/*
bias + relu
*/
template <typename T>
void PostAddBiasRelu(void* dst, const float* bias, long area, long oc4) {
    Float4 vzero(0.f);
    for (long z = oc4 - 1; z >= 0; --z) {
        Float4 vbias = Float4::load(bias + 4 * z);
        auto dst_z   = reinterpret_cast<T*>(dst) + area * 4 * z;
        long p       = 0;
        for (; p < area - 3; p += 4) {
            auto dst_p = dst_z + 4 * p;
            Float4x4 v = Float4x4::load(dst_p);
            v          = Float4x4::max(v + vbias, vzero);
            Float4x4::save(dst_p, v);
        }
        for (; p < area; ++p) {
            auto dst_p = dst_z + 4 * p;
            Float4::save(dst_p, Float4::max(Float4::load(dst_p) + vbias, vzero));
        }
    }
}
template void PostAddBiasRelu<float>(void* dst, const float* bias, long area, long oc4);
template void PostAddBiasRelu<bfp16_t>(void* dst, const float* bias, long area, long oc4);

/*
bias + relu6
*/
template <typename T>
void PostAddBiasRelu6(void* dst, const float* bias, long area, long oc4) {
    Float4 vzero(0.f);
    Float4 vrelu6(6.f);
    for (long z = oc4 - 1; z >= 0; --z) {
        Float4 vbias = Float4::load(bias + 4 * z);
        auto dst_z   = reinterpret_cast<T*>(dst) + area * 4 * z;
        long p       = 0;
        for (; p < area - 3; p += 4) {
            auto dst_p = dst_z + 4 * p;
            Float4x4 v = Float4x4::load(dst_p);
            v          = Float4x4::min(Float4x4::max(v + vbias, vzero), vrelu6);
            Float4x4::save(dst_p, v);
        }
        for (; p < area; ++p) {
            auto dst_p = dst_z + 4 * p;
            Float4::save(dst_p, Float4::min(Float4::max(Float4::load(dst_p) + vbias, vzero), vrelu6));
        }
    }
}
template void PostAddBiasRelu6<float>(void* dst, const float* bias, long area, long oc4);
template void PostAddBiasRelu6<bfp16_t>(void* dst, const float* bias, long area, long oc4);

/*
min(x, clap)
*/
template <typename T>
void PostClap(void* dst, long size4, float val) {
    Float4 vclap(val);
    long i = 0;
    for (; i < size4 - 3; i += 4) {
        auto dst_p = reinterpret_cast<T*>(dst) + 4 * i;
        Float4x4 v = Float4x4::load(dst_p);
        v          = Float4x4::min(v, vclap);
        Float4x4::save(dst_p, v);
    }
    for (; i < size4; i++) {
        Float4::save(reinterpret_cast<T*>(dst) + 4 * i,
                     Float4::min(Float4::load(reinterpret_cast<T*>(dst) + 4 * i), vclap));
    }
}
template void PostClap<float>(void* dst, long size4, float val);
template void PostClap<bfp16_t>(void* dst, long size4, float val);

#ifndef TNN_USE_NEON
/*
kernel func used in linux debug mode
*/

/*
general conv micro kernel
*/
template <typename T>
void ConvCommonO4(T* dst, const T* src, const float* weight, long width, long src_w_setup, long src_depth_quad,
                  long src_depth_step, long fw, long fh, long dilate_x_step, long dilate_y_step) {
    long dx, sz, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        T* dst_x             = dst + dx * 4;
        float dst_x_float[4] = {0};
        auto src_dx          = src + src_w_setup * dx;
        for (sz = 0; sz < src_depth_quad; ++sz) {
            auto src_z    = src_dx + sz * src_depth_step;
            auto weight_z = weight + sz * fh * fw * 16;
            for (fy = 0; fy < fh; ++fy) {
                auto src_y    = src_z + fy * dilate_y_step;
                auto weight_y = weight_z + fy * fw * 16;
                for (fx = 0; fx < fw; ++fx) {
                    auto weight_x = weight_y + 16 * fx;
                    auto src_x    = src_y + fx * dilate_x_step;
                    for (long i = 0; i < 4; ++i) {
                        for (long j = 0; j < 4; ++j) {
                            dst_x_float[j] += float(src_x[i]) * float(weight_x[4 * i + j]);
                        }
                    }
                }
            }
        }
        dst_x[0] = dst_x_float[0];
        dst_x[1] = dst_x_float[1];
        dst_x[2] = dst_x_float[2];
        dst_x[3] = dst_x_float[3];
    }
}

template void ConvCommonO4(float* dst, const float* src, const float* weight, long width, long src_w_setup,
                           long src_depth_quad, long src_depth_step, long fw, long fh, long dilate_x_step,
                           long dilate_y_step);

template void ConvCommonO4(bfp16_t* dst, const bfp16_t* src, const float* weight, long width, long src_w_setup,
                           long src_depth_quad, long src_depth_step, long fw, long fh, long dilate_x_step,
                           long dilate_y_step);

/*
general deconv micro kernel
*/
void DeconvFloatO4(float* dst, const float* src, const float* weight, long width, long dst_w_step, long src_depth_quad,
                   long src_depth_step, long fw, long fh, long dilate_x_step, long dilate_y_step) {
    long dx, sz, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        auto dst_dx = dst + dx * dst_w_step;
        for (fy = 0; fy < fh; ++fy) {
            auto dst_y    = dst_dx + fy * dilate_y_step;
            auto weight_y = weight + fy * fw * src_depth_quad * 16;
            for (fx = 0; fx < fw; ++fx) {
                auto dst_x    = dst_y + fx * dilate_x_step;
                auto weight_x = weight_y + fx * src_depth_quad * 16;
                float temp[4] = {0.f};
                for (sz = 0; sz < src_depth_quad; ++sz) {
                    auto weight_z = weight_x + sz * 16;
                    auto src_z    = src + dx * 4 + sz * src_depth_step;
                    for (long i = 0; i < 4; ++i) {
                        for (long j = 0; j < 4; ++j) {
                            temp[j] = temp[j] + float(src_z[i]) * weight_z[4 * i + j];
                        }
                    }
                }
                for (long j = 0; j < 4; ++j) {
                    dst_x[j] = float(dst_x[j]) + temp[j];
                }
            }
        }
    }
}

/*
micro kernel used in conv c3
*/
template <typename T>
void GemmSlidew(T* dst, const T* src, const float* weight, long width, long src_w_setup, long fw, long fh,
                long dilate_x_step, long dilate_y_step) {
    long dx, sz, fx, fy;
    for (dx = 0; dx < width; ++dx) {
        auto dst_x  = dst + dx * 4;
        dst_x[0]    = 0.0f;
        dst_x[1]    = 0.0f;
        dst_x[2]    = 0.0f;
        dst_x[3]    = 0.0f;
        auto src_dx = src + src_w_setup * dx;

        for (fy = 0; fy < fh; ++fy) {
            auto src_y    = src_dx + fy * dilate_y_step;
            auto weight_y = weight + fy * fw * 12;
            for (fx = 0; fx < fw; ++fx) {
                auto weight_x = weight_y + 12 * fx;
                auto src_x    = src_y + fx * dilate_x_step;
                for (long i = 0; i < 3; ++i) {
                    for (long j = 0; j < 4; ++j) {
                        dst_x[j] = float(dst_x[j]) + float(src_x[i]) * float(weight_x[4 * i + j]);
                    }
                }
            }
        }
    }
}

void GemmFloatSlidewC3(float* dst, const float* src, const float* weight, long width, long src_w_setup, long fw,
                       long fh, long dilate_x_step, long dilate_y_step) {
    GemmSlidew(dst, src, weight, width, src_w_setup, fw, fh, dilate_x_step, dilate_y_step);
}

void GemmBfp16SlidewC3(bfp16_t* dst, const bfp16_t* src, const float* weight, long width, long src_w_setup, long fw,
                       long fh, long dilate_x_step, long dilate_y_step) {
    GemmSlidew(dst, src, weight, width, src_w_setup, fw, fh, dilate_x_step, dilate_y_step);
}

/*
micro kernel used in convdw s1
*/
template <typename T>
void ConvDwSlideW(T* dst_z, T** cache_line, const float* weight_z, long dst_width, long fh, long fw) {
    long dx, fy, fx;
    for (dx = 0; dx < dst_width; ++dx) {
        auto dst_x = dst_z + dx * 4;
        dst_x[0]   = 0.0f;
        dst_x[1]   = 0.0f;
        dst_x[2]   = 0.0f;
        dst_x[3]   = 0.0f;
        for (fy = 0; fy < fh; ++fy) {
            for (fx = 0; fx < fw; ++fx) {
                for (long i = 0; i < 4; ++i) {
                    dst_x[i] = dst_x[i] + cache_line[fy][dx * 4 + fx * 4 + i] * weight_z[fw * fy * 4 + fx * 4 + i];
                }
            }
        }
    }
}

void ConvDw3x3FloatSlideW(void* dst_z, void** cache_line, const void* weight_z, long dst_width) {
    ConvDwSlideW(reinterpret_cast<float*>(dst_z), reinterpret_cast<float**>(cache_line),
                 reinterpret_cast<const float*>(weight_z), dst_width, 3, 3);
}
void ConvDw3x3Bfp16SlideW(void* dst_z, void** cache_line, const void* weight_z, long dst_width) {
    ConvDwSlideW(reinterpret_cast<bfp16_t*>(dst_z), reinterpret_cast<bfp16_t**>(cache_line),
                 reinterpret_cast<const float*>(weight_z), dst_width, 3, 3);
}
void ConvDw5x5FloatSlideW(void* dst_z, void** cache_line, const void* weight_z, long dst_width) {
    ConvDwSlideW(reinterpret_cast<float*>(dst_z), reinterpret_cast<float**>(cache_line),
                 reinterpret_cast<const float*>(weight_z), dst_width, 5, 5);
}
void ConvDw5x5Bfp16SlideW(void* dst_z, void** cache_line, const void* weight_z, long dst_width) {
    ConvDwSlideW(reinterpret_cast<bfp16_t*>(dst_z), reinterpret_cast<bfp16_t**>(cache_line),
                 reinterpret_cast<const float*>(weight_z), dst_width, 5, 5);
}

template <typename T>
void ActiveOutput(T* dst, const Float4* src, long relu, int num) {
    for (long i = 0; i < num; i++) {
        if (relu) {
            Float4::save(dst + i * 4, Float4::max(src[i], Float4(0.f)));
        } else {
            Float4::save(dst + i * 4, src[i]);
        }
    }
}

/*
micro kernel used in gemm like conv, such as conv1x1, conv3x3 winograd
*/
template <typename T>
void GEMM_FLOAT_NCHW(T* dst, const T* src, const float* weight, long src_depth_quad, long dst_step, long dst_depth_quad,
                     long width, float* bias, long relu) {
    long dx, sz, dz;
    long src_z_step = width * 4;
    for (dz = 0; dz < dst_depth_quad; ++dz) {
        auto dst_z     = dst + dz * dst_step;
        auto weight_dz = weight + dz * (src_depth_quad * 16);
        Float4 v_bias  = Float4::load(bias + dz * 4);
        // process 8x4 results in one loop
        for (dx = 0; dx + 7 < width; dx += 8) {
            auto dst_dx = dst_z + dx * 4;
            auto src_dx = src + dx * 4;
            Float4 v_dst[8];
            for (long i = 0; i < 8; i++)
                v_dst[i] = v_bias;
            for (long sz = 0; sz < src_depth_quad; ++sz) {
                auto src_z     = src_dx + sz * src_z_step;
                auto weight_sz = weight_dz + sz * 16;
                Float4 v_weight[4];
                for (long i = 0; i < 4; i++)
                    v_weight[i] = Float4::load(weight_sz + 4 * i);
                Float4 v_src[4][2];
                for (long i = 0; i < 4; i++) {
                    for (long j = 0; j < 2; j++) {
                        v_src[i][j] = Float4::load(src_z + i * 8 + j * 4);
                    }
                }
                for (long i = 0; i < 4; i++) {
                    v_dst[0] = v_dst[0] + v_weight[i] * v_src[i][0].value[0];
                    v_dst[1] = v_dst[1] + v_weight[i] * v_src[i][0].value[1];
                    v_dst[2] = v_dst[2] + v_weight[i] * v_src[i][0].value[2];
                    v_dst[3] = v_dst[3] + v_weight[i] * v_src[i][0].value[3];
                    v_dst[4] = v_dst[4] + v_weight[i] * v_src[i][1].value[0];
                    v_dst[5] = v_dst[5] + v_weight[i] * v_src[i][1].value[1];
                    v_dst[6] = v_dst[6] + v_weight[i] * v_src[i][1].value[2];
                    v_dst[7] = v_dst[7] + v_weight[i] * v_src[i][1].value[3];
                }
            }
            ActiveOutput(dst_dx, v_dst, relu, 8);
        }
        // process 4x4 results in one loop
        for (; dx + 3 < width; dx += 4) {
            auto dst_dx = dst_z + dx * 4;
            auto src_dx = src + dx * 4;
            Float4 v_dst[4];
            for (long i = 0; i < 4; i++)
                v_dst[i] = v_bias;
            for (long sz = 0; sz < src_depth_quad; ++sz) {
                auto src_z     = src_dx + sz * src_z_step;
                auto weight_sz = weight_dz + sz * 16;
                Float4 v_weight[4];
                for (long i = 0; i < 4; i++)
                    v_weight[i] = Float4::load(weight_sz + 4 * i);
                Float4 v_src[4];
                for (long i = 0; i < 4; i++)
                    v_src[i] = Float4::load(src_z + i * 4);

                for (long i = 0; i < 4; i++) {
                    v_dst[0] = v_dst[0] + v_weight[i] * v_src[i].value[0];
                    v_dst[1] = v_dst[1] + v_weight[i] * v_src[i].value[1];
                    v_dst[2] = v_dst[2] + v_weight[i] * v_src[i].value[2];
                    v_dst[3] = v_dst[3] + v_weight[i] * v_src[i].value[3];
                }
            }
            ActiveOutput(dst_dx, v_dst, relu, 4);
        }
        // the process 1x4 results in one loop
        for (; dx < width; ++dx) {
            auto dst_dx  = dst_z + dx * 4;
            auto src_dx  = src + dx * 4;
            Float4 v_dst = v_bias;
            for (long sz = 0; sz < src_depth_quad; ++sz) {
                auto src_z     = src_dx + sz * src_z_step;
                auto weight_sz = weight_dz + sz * 16;
                Float4 v_weight[4];
                for (long i = 0; i < 4; i++)
                    v_weight[i] = Float4::load(weight_sz + 4 * i);
                Float4 v_src = Float4::load(src_z);

                for (long i = 0; i < 4; i++) {
                    v_dst = v_dst + v_weight[i] * v_src.value[i];
                }
            }
            ActiveOutput(dst_dx, &v_dst, relu, 1);
        }
    }
}

void GEMM_BFP16_N4(bfp16_t* dst, const bfp16_t* src, const float* weight, long src_depth_quad, long dst_step,
                   long dst_depth_quad, long width, float* bias, long relu) {
    GEMM_FLOAT_NCHW(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, width, bias, relu);
}

void GEMM_FLOAT_N4(float* dst, const float* src, const float* weight, long src_depth_quad, long dst_step,
                   long dst_depth_quad, long width, float* bias, long relu) {
    GEMM_FLOAT_NCHW(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, width, bias, relu);
}
#endif

#ifdef TNN_USE_NEON
/*
assemble arm neon kernel, used in conv common
*/
template <>
void ConvCommonO4(float* dst, const float* src, const float* weight, long width, long src_w_step, long src_depth_quad,
                  long src_depth_step, long fw, long fh, long dilate_x_step, long dilate_y_step) {
    ConvFloatO4(dst, src, weight, width, src_w_step, src_depth_quad, src_depth_step, fw, fh, dilate_x_step,
                dilate_y_step);
}

template <>
void ConvCommonO4(bfp16_t* dst, const bfp16_t* src, const float* weight, long width, long src_w_step,
                  long src_depth_quad, long src_depth_step, long fw, long fh, long dilate_x_step, long dilate_y_step) {
    ConvBfp16O4(dst, src, weight, width, src_w_step, src_depth_quad, src_depth_step, fw, fh, dilate_x_step,
                dilate_y_step);
}
#endif

/*
max pooling corner func, left/right/top/bottom
*/
template <typename T>
void MaxPoolingCorner(const T* src, long iw, long ih, T* dst, long ow, long kw, long kh, long stride_w, long stride_h,
                      long pad_w, long pad_h, long l, long r, long t, long b) {
    for (long oy = t; oy < b; ++oy) {
        for (long ox = l; ox < r; ++ox) {
            Float4 vmax(-FLT_MAX);

            const long srcOriginX = ox * stride_w - pad_w;
            const long srcOriginY = oy * stride_h - pad_h;
            const long kxs        = MAX(0, -srcOriginX);
            const long kxe        = MIN(kw, iw - srcOriginX);
            const long kys        = MAX(0, -srcOriginY);
            const long kye        = MIN(kh, ih - srcOriginY);
            const auto src_ptr    = src + (srcOriginY * iw + srcOriginX) * 4;
            auto dst_ptr          = dst + (oy * ow + ox) * 4;

            for (long ky = kys; ky < kye; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * 4;
                for (long kx = kxs; kx < kxe; kx++) {
                    vmax = Float4::max(vmax, Float4::load(src_ptr_h + kx * 4));
                }
            }

            Float4::save(dst_ptr, vmax);
        }
    }
}

template void MaxPoolingCorner(const float* src, long iw, long ih, float* dst, long ow, long kw, long kh, long stride_w,
                               long stride_h, long pad_w, long pad_h, long l, long r, long t, long b);

template void MaxPoolingCorner(const bfp16_t* src, long iw, long ih, bfp16_t* dst, long ow, long kw, long kh,
                               long stride_w, long stride_h, long pad_w, long pad_h, long l, long r, long t, long b);

/*
max pooling 3x3s2 kernel
*/
template <typename T>
void MaxPoolingCenter3x3s2(const T* src, long iw, long ih, T* dst, long ow, long oh, long pad_w, long pad_h, long l,
                           long r, long t, long b) {
    for (long oy = t; oy < b; ++oy) {
        for (long ox = l; ox < r; ++ox) {
            Float4 vmax(-FLT_MAX);

            const long src_offset_x = ox * 2 - pad_w;
            const long src_offset_y = oy * 2 - pad_h;
            const auto src_ptr      = src + (src_offset_y * iw + src_offset_x) * 4;
            auto dst_ptr            = dst + (oy * ow + ox) * 4;

            for (long ky = 0; ky < 3; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * 4;
                vmax                 = Float4::max(vmax, Float4::load(src_ptr_h + 0 * 4));
                vmax                 = Float4::max(vmax, Float4::load(src_ptr_h + 1 * 4));
                vmax                 = Float4::max(vmax, Float4::load(src_ptr_h + 2 * 4));
            }
            Float4::save(dst_ptr, vmax);
        }
    }
}

template void MaxPoolingCenter3x3s2(const float* src, long iw, long ih, float* dst, long ow, long oh, long pad_w,
                                    long pad_h, long l, long r, long t, long b);
template void MaxPoolingCenter3x3s2(const bfp16_t* src, long iw, long ih, bfp16_t* dst, long ow, long oh, long pad_w,
                                    long pad_h, long l, long r, long t, long b);

/*
general max pooling center kernel
*/
template <typename T>
void MaxPoolingCenter(const T* src, long iw, long ih, T* dst, long ow, long oh, long kw, long kh, long stride_w,
                      long stride_h, long pad_w, long pad_h, long l, long r, long t, long b) {
    for (long oy = t; oy < b; ++oy) {
        for (long ox = l; ox < r; ++ox) {
            Float4 vmax(-FLT_MAX);

            const long src_offset_x = ox * stride_w - pad_w;
            const long src_offset_y = oy * stride_h - pad_h;
            const auto src_ptr      = src + (src_offset_y * iw + src_offset_x) * 4;
            auto dst_ptr            = dst + (oy * ow + ox) * 4;

            for (long ky = 0; ky < kh; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * 4;
                for (long kx = 0; kx < kw; kx++) {
                    vmax = Float4::max(vmax, Float4::load(src_ptr_h + kx * 4));
                }
            }

            Float4::save(dst_ptr, vmax);
        }
    }
}

/*
max pooling func, process four corners and center
*/
template <typename T>
void MaxPooling(const T* src, long iw, long ih, T* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h, long l, long r, long t, long b) {
    // top corner
    MaxPoolingCorner<T>(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, 0, ow, 0, t);
    if (kw == 3 && kh == 3 && stride_h == 2 && stride_w == 2) {
        MaxPoolingCenter3x3s2<T>(src, iw, ih, dst, ow, oh, pad_w, pad_h, l, r, t, b);
    } else {
        MaxPoolingCenter<T>(src, iw, ih, dst, ow, oh, kw, kh, stride_w, stride_h, pad_w, pad_h, l, r, t, b);
    }

    // bottom corner
    MaxPoolingCorner<T>(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, 0, ow, b, oh);
    // left corner
    MaxPoolingCorner<T>(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, 0, l, t, b);
    // right corner
    MaxPoolingCorner<T>(src, iw, ih, dst, ow, kw, kh, stride_w, stride_h, pad_w, pad_h, r, ow, t, b);
}
template void MaxPooling(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh,
                         long stride_w, long stride_h, long pad_w, long pad_h, long l, long r, long t, long b);
template void MaxPooling(const bfp16_t* src, long iw, long ih, bfp16_t* dst, long ow, long oh, long kw, long kh,
                         long stride_w, long stride_h, long pad_w, long pad_h, long l, long r, long t, long b);

/*
general avg pooling func
*/
template <typename T>
void AvgPooling(const T* src, long iw, long ih, T* dst, long ow, long oh, long kw, long kh, long stride_w,
                long stride_h, long pad_w, long pad_h) {
    for (long oy = 0; oy < oh; ++oy) {
        for (long ox = 0; ox < ow; ++ox) {
            Float4 vavg(0.f);

            const long srcOriginX    = ox * stride_w - pad_w;
            const long srcOriginY    = oy * stride_h - pad_h;
            const long kxs           = MAX(0, -srcOriginX);
            const long kxe           = MIN(kw, iw - srcOriginX);
            const long kys           = MAX(0, -srcOriginY);
            const long kye           = MIN(kh, ih - srcOriginY);
            const float kernel_count = 1.0 / ((kxe - kxs) * (kye - kys));
            const auto src_ptr       = src + (srcOriginY * iw + srcOriginX) * 4;
            auto dst_ptr             = dst + (oy * ow + ox) * 4;

            for (long ky = kys; ky < kye; ++ky) {
                const auto src_ptr_h = src_ptr + (ky * iw) * 4;
                for (long kx = kxs; kx < kxe; kx++) {
                    vavg = vavg + Float4::load(src_ptr_h + kx * 4);
                }
            }

            vavg = vavg * Float4(kernel_count);
            Float4::save(dst_ptr, vavg);
        }
    }
}

template void AvgPooling(const float* src, long iw, long ih, float* dst, long ow, long oh, long kw, long kh,
                         long stride_w, long stride_h, long pad_w, long pad_h);
template void AvgPooling(const bfp16_t* src, long iw, long ih, bfp16_t* dst, long ow, long oh, long kw, long kh,
                         long stride_w, long stride_h, long pad_w, long pad_h);

/*
convdw unit, used in four cornels calc
*/
template <typename T>
void DepthwiseUnit(T* dst, const T* src, const float* weight, long fw, long fh, long weight_y_step, long dilate_x_step,
                   long dilate_y_step) {
    long fx, fy;
    Float4 dst_v(0.0f);
    const auto* src_z    = src;
    const auto* weight_z = weight;
    for (fy = 0; fy < fh; ++fy) {
        const auto* src_y    = src_z + fy * dilate_y_step;
        const auto* weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            Float4 src_v    = Float4::load(src_y + fx * dilate_x_step);
            Float4 weight_v = Float4::load(weight_y + 4 * fx);
            Float4::mla(dst_v, src_v, weight_v);
        }
    }
    Float4::save(dst, dst_v);
}
template void DepthwiseUnit(float* dst, const float* src, const float* weight, long fw, long fh, long weight_y_step,
                            long dilate_x_step, long dilate_y_step);
template void DepthwiseUnit(bfp16_t* dst, const bfp16_t* src, const float* weight, long fw, long fh, long weight_y_step,
                            long dilate_x_step, long dilate_y_step);

/*
general convdw func
*/
template <typename T>
void DepthwiseConv(T* dst, const T* src, const float* weight, long width, long src_w_step, long fw, long fh,
                   long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep) {
    long dx, fx, fy;
    for (long y = 0; y < height; ++y) {
        auto srcY = src + y * srcHStep;
        auto dstY = dst + y * dstHStep;
        dx        = 0;
        for (; dx + 3 < width; dx += 4) {
            Float4 dst_v[4];
            for (long i = 0; i < 4; i++)
                dst_v[i] = 0.0f;
            const auto* src_z    = srcY + src_w_step * dx;
            const auto* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const auto* src_y    = src_z + fy * dilate_y_step;
                const auto* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    Float4 weight_v = Float4::load(weight_y + 4 * fx);
                    Float4 src_v0   = Float4::load(src_y + fx * dilate_x_step);
                    Float4::mla(dst_v[0], src_v0, weight_v);
                    Float4 src_v1 = Float4::load(src_y + fx * dilate_x_step + src_w_step);
                    Float4::mla(dst_v[1], src_v1, weight_v);
                    Float4 src_v2 = Float4::load(src_y + fx * dilate_x_step + 2 * src_w_step);
                    Float4::mla(dst_v[2], src_v2, weight_v);
                    Float4 src_v3 = Float4::load(src_y + fx * dilate_x_step + 3 * src_w_step);
                    Float4::mla(dst_v[3], src_v3, weight_v);
                }
            }
            Float4::save(dstY + (dx + 0) * 4, dst_v[0]);
            Float4::save(dstY + (dx + 1) * 4, dst_v[1]);
            Float4::save(dstY + (dx + 2) * 4, dst_v[2]);
            Float4::save(dstY + (dx + 3) * 4, dst_v[3]);
        }
        for (; dx < width; ++dx) {
            Float4 dst_v(0.0f);
            const auto* src_z    = srcY + src_w_step * dx;
            const auto* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const auto* src_y    = src_z + fy * dilate_y_step;
                const auto* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    Float4 src_v    = Float4::load(src_y + fx * dilate_x_step);
                    Float4 weight_v = Float4::load(weight_y + 4 * fx);
                    dst_v           = dst_v + src_v * weight_v;
                }
            }
            Float4::save(dstY + dx * 4, dst_v);
        }
    }
}
template void DepthwiseConv(float* dst, const float* src, const float* weight, long width, long src_w_step, long fw,
                            long fh, long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep);
template void DepthwiseConv(bfp16_t* dst, const bfp16_t* src, const float* weight, long width, long src_w_step, long fw,
                            long fh, long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep);

/*
convdw3x3 center func
*/
template <typename T>
void DepthwiseConv3x3(T* dst, const T* src, const float* weight, long width, long src_w_step, long fw, long fh,
                      long dilate_x_step, long dilate_y_step, long height, long srcHStep, long dstHStep) {
    long dx, fx, fy;
    Float4 weight_v[9];
    for (long i = 0; i < 9; i++)
        weight_v[i] = Float4::load(weight + i * 4);

    for (long y = 0; y < height; ++y) {
        auto srcY = src + y * srcHStep;
        auto dstY = dst + y * dstHStep;
        dx        = 0;
        for (; dx + 3 < width; dx += 4) {
            Float4 dst_v[4];
            for (long i = 0; i < 4; i++)
                dst_v[i] = 0.0f;
            const auto* src_z = srcY + src_w_step * dx;
            for (fy = 0; fy < 3; ++fy) {
                const auto* src_y = src_z + fy * dilate_y_step;
                for (fx = 0; fx < 3; ++fx) {
                    Float4 src_v0 = Float4::load(src_y + fx * dilate_x_step);
                    Float4::mla(dst_v[0], src_v0, weight_v[fy * 3 + fx]);
                    Float4 src_v1 = Float4::load(src_y + fx * dilate_x_step + src_w_step);
                    Float4::mla(dst_v[1], src_v1, weight_v[fy * 3 + fx]);
                    Float4 src_v2 = Float4::load(src_y + fx * dilate_x_step + 2 * src_w_step);
                    Float4::mla(dst_v[2], src_v2, weight_v[fy * 3 + fx]);
                    Float4 src_v3 = Float4::load(src_y + fx * dilate_x_step + 3 * src_w_step);
                    Float4::mla(dst_v[3], src_v3, weight_v[fy * 3 + fx]);
                }
            }
            Float4::save(dstY + (dx + 0) * 4, dst_v[0]);
            Float4::save(dstY + (dx + 1) * 4, dst_v[1]);
            Float4::save(dstY + (dx + 2) * 4, dst_v[2]);
            Float4::save(dstY + (dx + 3) * 4, dst_v[3]);
        }
        for (; dx < width; ++dx) {
            Float4 dst_v(0.0f);
            const auto* src_z    = srcY + src_w_step * dx;
            const auto* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const auto* src_y    = src_z + fy * dilate_y_step;
                const auto* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    Float4 src_v    = Float4::load(src_y + fx * dilate_x_step);
                    Float4 weight_v = Float4::load(weight_y + 4 * fx);
                    dst_v           = dst_v + src_v * weight_v;
                }
            }
            Float4::save(dstY + dx * 4, dst_v);
        }
    }
}

template void DepthwiseConv3x3(float* dst, const float* src, const float* weight, long width, long src_w_step, long fw,
                               long fh, long dilate_x_step, long dilate_y_step, long height, long srcHStep,
                               long dstHStep);
template void DepthwiseConv3x3(bfp16_t* dst, const bfp16_t* src, const float* weight, long width, long src_w_step,
                               long fw, long fh, long dilate_x_step, long dilate_y_step, long height, long srcHStep,
                               long dstHStep);

template <typename Tin, typename Tout>
void FloatConvert(const Tin* src, Tout* dst, long area_quad) {
    // need support inplace
    int i = area_quad - 1;
    // for (; i >= 3; i -= 4) {
    //     Float4x4::save(dst + i * 16, Float4x4::load(src + i * 16));
    // }
    for (; i >= 0; --i) {
        Float4::save(dst + i * 4, Float4::load(src + i * 4));
    }
}

/*
data convert between bfp16 and float32
*/
template void FloatConvert(const float* src, bfp16_t* dst, long area_quad);
template void FloatConvert(const bfp16_t* src, float* dst, long area_quad);

/*
deconv dw unit
*/
template <typename T>
void DepthwiseUnitDeconv(const T* dst, T* src, const float* weight, long fw, long fh, long weight_y_step,
                         long dilate_x_step, long dilate_y_step) {
    long fx, fy;
    T* src_z              = src;
    const float* weight_z = weight;
    Float4 dstV           = Float4::load(dst);
    for (fy = 0; fy < fh; ++fy) {
        T* src_y              = src_z + fy * dilate_y_step;
        const float* weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            Float4 weight_x = Float4::load(weight_y + 4 * fx);
            Float4 src_x    = Float4::load(src_y + fx * dilate_x_step);
            Float4::save(src_y + fx * dilate_x_step, src_x + weight_x * dstV);
        }
    }
}

template void DepthwiseUnitDeconv(const float* dst, float* src, const float* weight, long fw, long fh,
                                  long weight_y_step, long dilate_x_step, long dilate_y_step);
template void DepthwiseUnitDeconv(const bfp16_t* dst, bfp16_t* src, const float* weight, long fw, long fh,
                                  long weight_y_step, long dilate_x_step, long dilate_y_step);

/*
general deconv dw func
*/
template <typename T>
void DepthwiseDeconv(const T* dst, T* src, const float* weight, long width, long src_w_setup, long fw, long fh,
                     long dilate_x_step, long dilate_y_step) {
    long dx;
    for (dx = 0; dx < width; ++dx) {
        const T* dst_x = dst + dx * 4;
        T* src_dx      = src + src_w_setup * dx;
        DepthwiseUnitDeconv(dst_x, src_dx, weight, fw, fh, fw * 4, dilate_x_step, dilate_y_step);
    }
}

template void DepthwiseDeconv(const float* dst, float* src, const float* weight, long width, long src_w_setup, long fw,
                              long fh, long dilate_x_step, long dilate_y_step);
template void DepthwiseDeconv(const bfp16_t* dst, bfp16_t* src, const float* weight, long width, long src_w_setup,
                              long fw, long fh, long dilate_x_step, long dilate_y_step);

}  // namespace TNN_NS
