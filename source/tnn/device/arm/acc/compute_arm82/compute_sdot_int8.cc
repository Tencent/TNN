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

#include "tnn/device/arm/acc/compute_arm82/compute_sdot_int8.h"
#include "tnn/utils/naive_compute.h"

#define SIMU_SDOT_8X8()                                              \
    dst_v[j * 8 + 0] += (int32_t)src_v * (int32_t)weight_i[k + 0];   \
    dst_v[j * 8 + 1] += (int32_t)src_v * (int32_t)weight_i[k + 4];   \
    dst_v[j * 8 + 2] += (int32_t)src_v * (int32_t)weight_i[k + 8];   \
    dst_v[j * 8 + 3] += (int32_t)src_v * (int32_t)weight_i[k + 12];  \
    dst_v[j * 8 + 4] += (int32_t)src_v * (int32_t)weight_i[k + 16];  \
    dst_v[j * 8 + 5] += (int32_t)src_v * (int32_t)weight_i[k + 20];  \
    dst_v[j * 8 + 6] += (int32_t)src_v * (int32_t)weight_i[k + 24];  \
    dst_v[j * 8 + 7] += (int32_t)src_v * (int32_t)weight_i[k + 28];

#define SIMU_SDOT_1X8()                                              \
    dst_v[0] += (int32_t)src_v * (int32_t)weight_i[k + 0];           \
    dst_v[1] += (int32_t)src_v * (int32_t)weight_i[k + 4];           \
    dst_v[2] += (int32_t)src_v * (int32_t)weight_i[k + 8];           \
    dst_v[3] += (int32_t)src_v * (int32_t)weight_i[k + 12];          \
    dst_v[4] += (int32_t)src_v * (int32_t)weight_i[k + 16];          \
    dst_v[5] += (int32_t)src_v * (int32_t)weight_i[k + 20];          \
    dst_v[6] += (int32_t)src_v * (int32_t)weight_i[k + 24];          \
    dst_v[7] += (int32_t)src_v * (int32_t)weight_i[k + 28];

#define SIMU_SDOT_8X4()                                              \
    dst_v[j * 4 + 0] += (int32_t)src_v * (int32_t)weight_i[k + 0];   \
    dst_v[j * 4 + 1] += (int32_t)src_v * (int32_t)weight_i[k + 4];   \
    dst_v[j * 4 + 2] += (int32_t)src_v * (int32_t)weight_i[k + 8];   \
    dst_v[j * 4 + 3] += (int32_t)src_v * (int32_t)weight_i[k + 12];

#define SIMU_SDOT_1X4()                                              \
    dst_v[0] += (int32_t)src_v * (int32_t)weight_i[k + 0];           \
    dst_v[1] += (int32_t)src_v * (int32_t)weight_i[k + 4];           \
    dst_v[2] += (int32_t)src_v * (int32_t)weight_i[k + 8];           \
    dst_v[3] += (int32_t)src_v * (int32_t)weight_i[k + 12];

namespace TNN_NS {

void GEMM_SDOT_INT8_8X8(int8_t* dst, const int8_t* src, const int8_t* weight, long src_depth,
                       long dst_depth, long hw, const int32_t* bias, const float* scale,
                       long relu, const int8_t* add_input, const float* add_scale,
                       const int8_t* relu6_max) {
#ifdef TNN_ARM82_A64
    GemmInt8SdotUnit8x8(dst, src, weight, src_depth, dst_depth, hw, bias, scale,
                        relu, add_input, add_scale, relu6_max);
#else
    // src_depth round_up with 4
    // dst_depth round_up with 4
    // NHWC4 src & dst
    // oc8
    for (long dz = 0; dz + 7 < dst_depth; dz += 8) {
        auto dst_z = dst + dz;
        auto weight_z = weight + dz * src_depth;
        auto bias_z = bias + dz;
        auto scale_z = scale + dz;

        // process hw8 x oc8 results = 
        // src [hw8, depth] * weight [oc8, depth]
        const int8_t* src_ptr[8];
        long dx = 0;
        // hw8
        for (; dx + 7 < hw; dx += 8) {
            auto dst_dx = dst_z + dx * dst_depth;
            src_ptr[0] = src + dx * src_depth;
            src_ptr[1] = src_ptr[0] + src_depth;
            src_ptr[2] = src_ptr[1] + src_depth;
            src_ptr[3] = src_ptr[2] + src_depth;
            src_ptr[4] = src_ptr[3] + src_depth;
            src_ptr[5] = src_ptr[4] + src_depth;
            src_ptr[6] = src_ptr[5] + src_depth;
            src_ptr[7] = src_ptr[6] + src_depth;
            int32_t dst_v[64];
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    dst_v[i * 8 + j] = bias_z[j];
                }
            }
            // crr
            long sz = 0;
            for (; sz + 15 < src_depth; sz += 16) {
                auto weight_sz = weight_z + sz * 8;
                // oc0:c0,c1,c2,c3 | oc1:c0,c1,c2,c3 | ... | oc7:c0,c1,c2,c3
                // oc0:c4,c5,c6,c7 | oc1:c4,c5,c6,c7 | ... | oc7:c4,c5,c6,c7
                // oc0:c8,c9,c10,c11 | ... | oc7:c8,c9,c10,c11
                // oc0:c12,c13,c14,c15 | ... | oc7:c12,c13,c14,c15
                for (int i = 0; i < 4; i++) {
                    auto weight_i = weight_sz + i * 8 * 4;
                    for (int j = 0; j < 8; j++) {
                        for (int k = 0; k < 4; k++) {
                            // crr = i * 4 + k
                            auto src_v = src_ptr[j][sz + i * 4 + k];
                            SIMU_SDOT_8X8();
                        }
                    }
                }
            }
            for (; sz + 7 < src_depth; sz += 8) {
                auto weight_sz = weight_z + sz * 8;
                // oc0:c0,c1,c2,c3 | oc1:c0,c1,c2,c3 | ... | oc7:c0,c1,c2,c3
                // oc0:c4,c5,c6,c7 | oc1:c4,c5,c6,c7 | ... | oc7:c4,c5,c6,c7
                for (int i = 0; i < 2; i++) {
                    auto weight_i = weight_sz + i * 8 * 4;
                    for (int j = 0; j < 8; j++) {
                        for (int k = 0; k < 4; k++) {
                            // crr = i * 4 + k
                            auto src_v = src_ptr[j][sz + i * 4 + k];
                            SIMU_SDOT_8X8();
                        }
                    }
                }
            }
            for (; sz < src_depth; sz += 4) {
                auto weight_sz = weight_z + sz * 8;
                auto weight_i = weight_sz;
                for (int j = 0; j < 8; j++) {
                    for (int k = 0; k < 4; k++) {
                        // crr = k
                        auto src_v = src_ptr[j][sz + k];
                        SIMU_SDOT_8X8();
                    }
                }
            }

            float res[64];
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    res[i * 8 + j] = dst_v[i * 8 + j] * scale_z[j];
                }
            }
            // Conv-Relu-Add
            if (relu == -1) {
                for (int i = 0; i < 64; i++) {
                    res[i] = MAX(0, res[i]);
                }
            }
            if (add_input) {
                auto add_input_ptr = add_input + dx * dst_depth + dz;
                auto add_scale_ptr = add_scale + dz;
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        res[i * 8 + j] += add_input_ptr[i * dst_depth + j] * add_scale_ptr[j];
                    }
                }
            }
            // Conv-Add-Relu
            if (relu == 1) {
                for (int i = 0; i < 64; i++) {
                    res[i] = MAX(0, res[i]);
                }
            }
            // Conv-Add-Relu6
            else if (relu == 2) {
                auto relu6_max_ptr = relu6_max + dz;
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        res[i * 8 + j] = MIN(res[i * 8 + j], (float)relu6_max_ptr[j]);
                        res[i * 8 + j] = MAX(res[i * 8 + j], 0);
                    }
                }
            }

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    dst_dx[i * dst_depth + j] = float2int8(res[i * 8 + j]);
                }
            }
        }
        // corner case
        // process hw x oc8 results = 
        // src [1, depth] * weight [oc8, depth]
        for (; dx < hw; dx++) {
            auto dst_dx = dst_z + dx * dst_depth;
            auto src_dx = src + dx * src_depth;
            int32_t dst_v[8];
            for (int i = 0; i < 8; i++) {
                dst_v[i] = bias_z[i];
            }
            // crr
            long sz = 0;
            for (; sz + 15 < src_depth; sz += 16) {
                auto weight_sz = weight_z + sz * 8;
                for (int i = 0; i < 4; i++) {
                    auto weight_i = weight_sz + i * 8 * 4;
                    for (int k = 0; k < 4; k++) {
                        auto src_v = src_dx[sz + i * 4 + k];
                        SIMU_SDOT_1X8();
                    }
                }
            }
            for (; sz + 7 < src_depth; sz += 8) {
                auto weight_sz = weight_z + sz * 8;
                for (int i = 0; i < 2; i++) {
                    auto weight_i = weight_sz + i * 8 * 4;
                    for (int k = 0; k < 4; k++) {
                        auto src_v = src_dx[sz + i * 4 + k];
                        SIMU_SDOT_1X8();
                    }
                }
            }
            for (; sz < src_depth; sz += 4) {
                auto weight_sz = weight_z + sz * 8;
                auto weight_i = weight_sz;
                for (int k = 0; k < 4; k++) {
                    auto src_v = src_dx[sz + k];
                    SIMU_SDOT_1X8();
                }
            }

            float res[8];
            for (int i = 0; i < 8; i++) {
                res[i] = dst_v[i] * scale_z[i];
            }
            // Conv-Relu-Add
            if (relu == -1) {
                for (int i = 0; i < 8; i++) {
                    res[i] = MAX(0, res[i]);
                }
            }
            if (add_input) {
                auto add_input_ptr = add_input + dx * dst_depth + dz;
                auto add_scale_ptr = add_scale + dz;
                for (int i = 0; i < 8; i++) {
                    res[i] += add_input_ptr[i] * add_scale_ptr[i];
                }
            }
            // Conv-Add-Relu
            if (relu == 1) {
                for (int i = 0; i < 8; i++) {
                    res[i] = MAX(0, res[i]);
                }
            }
            // Conv-Add-Relu6
            else if (relu == 2) {
                auto relu6_max_ptr = relu6_max + dz;
                for (int i = 0; i < 8; i++) {
                    res[i] = MIN(res[i], (float)relu6_max_ptr[i]);
                    res[i] = MAX(res[i], 0);
                }
            }

            for (int i = 0; i < 8; i++) {
                dst_dx[i] = float2int8(res[i]);
            }
        }
    }
#endif
}

void GEMM_SDOT_INT8_8X4(int8_t* dst, const int8_t* src, const int8_t* weight, long src_depth,
                       long dst_depth, long hw, const int32_t* bias, const float* scale,
                       long relu, const int8_t* add_input, const float* add_scale,
                       const int8_t* relu6_max) {
#ifdef TNN_ARM82_A64
    GemmInt8SdotUnit8x4(dst, src, weight, src_depth, dst_depth, hw, bias, scale,
                        relu, add_input, add_scale, relu6_max);
#else
    // src_depth round_up with 4
    // dst_depth round_up with 4
    // NHWC4 src & dst
    // corner case oc4
    int dz = 0;
    auto dst_z = dst + dz;
    auto weight_z = weight + dz * src_depth;
    auto bias_z = bias + dz;
    auto scale_z = scale + dz;

    // process hw8 x oc4 results = 
    // src [hw8, depth] * weight [oc4, depth]
    const int8_t* src_ptr[8];
    long dx = 0;
    // hw8
    for (; dx + 7 < hw; dx += 8) {
        auto dst_dx = dst_z + dx * dst_depth;
        src_ptr[0] = src + dx * src_depth;
        src_ptr[1] = src_ptr[0] + src_depth;
        src_ptr[2] = src_ptr[1] + src_depth;
        src_ptr[3] = src_ptr[2] + src_depth;
        src_ptr[4] = src_ptr[3] + src_depth;
        src_ptr[5] = src_ptr[4] + src_depth;
        src_ptr[6] = src_ptr[5] + src_depth;
        src_ptr[7] = src_ptr[6] + src_depth;
        int32_t dst_v[32];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 4; j++) {
                dst_v[i * 4 + j] = bias_z[j];
            }
        }
        // crr
        long sz = 0;
        for (; sz + 15 < src_depth; sz += 16) {
            auto weight_sz = weight_z + sz * 4;
            // oc0:c0,c1,c2,c3 | oc1:c0,c1,c2,c3 | ... | oc3:c0,c1,c2,c3
            // oc0:c4,c5,c6,c7 | oc1:c4,c5,c6,c7 | ... | oc3:c4,c5,c6,c7
            // oc0:c8,c9,c10,c11 | ... | oc3:c8,c9,c10,c11
            // oc0:c12,c13,c14,c15 | ... | oc3:c12,c13,c14,c15
            for (int i = 0; i < 4; i++) {
                auto weight_i = weight_sz + i * 4 * 4;
                for (int j = 0; j < 8; j++) {
                    for (int k = 0; k < 4; k++) {
                        // crr = i * 4 + k
                        auto src_v = src_ptr[j][sz + i * 4 + k];
                        SIMU_SDOT_8X4();
                    }
                }
            }
        }
        for (; sz + 7 < src_depth; sz += 8) {
            auto weight_sz = weight_z + sz * 4;
            // oc0:c0,c1,c2,c3 | oc1:c0,c1,c2,c3 | ... | oc3:c0,c1,c2,c3
            // oc0:c4,c5,c6,c7 | oc1:c4,c5,c6,c7 | ... | oc3:c4,c5,c6,c7
            for (int i = 0; i < 2; i++) {
                auto weight_i = weight_sz + i * 4 * 4;
                for (int j = 0; j < 8; j++) {
                    for (int k = 0; k < 4; k++) {
                        // crr = i * 4 + k
                        auto src_v = src_ptr[j][sz + i * 4 + k];
                        SIMU_SDOT_8X4();
                    }
                }
            }
        }
        for (; sz < src_depth; sz += 4) {
            auto weight_sz = weight_z + sz * 4;
            auto weight_i = weight_sz;
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 4; k++) {
                    // crr = k
                    auto src_v = src_ptr[j][sz + k];
                    SIMU_SDOT_8X4();
                }
            }
        }

        float res[32];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 4; j++) {
                res[i * 4 + j] = dst_v[i * 4 + j] * scale_z[j];
            }
        }
        // Conv-Relu-Add
        if (relu == -1) {
            for (int i = 0; i < 32; i++) {
                res[i] = MAX(0, res[i]);
            }
        }
        if (add_input) {
            auto add_input_ptr = add_input + dx * dst_depth + dz;
            auto add_scale_ptr = add_scale + dz;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 4; j++) {
                    res[i * 4 + j] += add_input_ptr[i * dst_depth + j] * add_scale_ptr[j];
                }
            }
        }
        // Conv-Add-Relu
        if (relu == 1) {
            for (int i = 0; i < 32; i++) {
                res[i] = MAX(0, res[i]);
            }
        }
        // Conv-Add-Relu6
        else if (relu == 2) {
            auto relu6_max_ptr = relu6_max + dz;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 4; j++) {
                    res[i * 4 + j] = MIN(res[i * 4 + j], (float)relu6_max_ptr[j]);
                    res[i * 4 + j] = MAX(res[i * 4 + j], 0);
                }
            }
        }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 4; j++) {
                dst_dx[i * dst_depth + j] = float2int8(res[i * 4 + j]);
            }
        }
    }
    // corner case
    // process hw x oc4 results = 
    // src [1, depth] * weight [oc4, depth]
    for (; dx < hw; dx++) {
        auto dst_dx = dst_z + dx * dst_depth;
        auto src_dx = src + dx * src_depth;
        int32_t dst_v[4];
        for (int i = 0; i < 4; i++) {
            dst_v[i] = bias_z[i];
        }
        // crr
        long sz = 0;
        for (; sz + 15 < src_depth; sz += 16) {
            auto weight_sz = weight_z + sz * 4;
            for (int i = 0; i < 4; i++) {
                auto weight_i = weight_sz + i * 4 * 4;
                for (int k = 0; k < 4; k++) {
                    auto src_v = src_dx[sz + i * 4 + k];
                    SIMU_SDOT_1X4();
                }
            }
        }
        for (; sz + 7 < src_depth; sz += 8) {
            auto weight_sz = weight_z + sz * 4;
            for (int i = 0; i < 2; i++) {
                auto weight_i = weight_sz + i * 4 * 4;
                for (int k = 0; k < 4; k++) {
                    auto src_v = src_dx[sz + i * 4 + k];
                    SIMU_SDOT_1X4();
                }
            }
        }
        for (; sz < src_depth; sz += 4) {
            auto weight_sz = weight_z + sz * 4;
            auto weight_i = weight_sz;
            for (int k = 0; k < 4; k++) {
                auto src_v = src_dx[sz + k];
                SIMU_SDOT_1X4();
            }
        }

        float res[4];
        for (int i = 0; i < 4; i++) {
            res[i] = dst_v[i] * scale_z[i];
        }
        // Conv-Relu-Add
        if (relu == -1) {
            for (int i = 0; i < 4; i++) {
                res[i] = MAX(0, res[i]);
            }
        }
        if (add_input) {
            auto add_input_ptr = add_input + dx * dst_depth + dz;
            auto add_scale_ptr = add_scale + dz;
            for (int i = 0; i < 4; i++) {
                res[i] += add_input_ptr[i] * add_scale_ptr[i];
            }
        }
        // Conv-Add-Relu
        if (relu == 1) {
            for (int i = 0; i < 4; i++) {
                res[i] = MAX(0, res[i]);
            }
        }
        // Conv-Add-Relu6
        else if (relu == 2) {
            auto relu6_max_ptr = relu6_max + dz;
            for (int i = 0; i < 4; i++) {
                res[i] = MIN(res[i], (float)relu6_max_ptr[i]);
                res[i] = MAX(res[i], 0);
            }
        }

        for (int i = 0; i < 4; i++) {
            dst_dx[i] = float2int8(res[i]);
        }
    }
#endif
}

void PackSDOTINT8Weight(const int8_t *src, int8_t *dst, int oc, int ic, int kh, int kw) {
    int oc_r4  = ROUND_UP(oc, 4);
    int ic_r4  = ROUND_UP(ic, 4);
    int crs    = ic * kw * kh;
    int crs_r4 = ic_r4 * kw * kh;
    int ic_align_4 = ROUND_UP(ic, 4);
    int oc_r4_align = oc_r4 / 8 * 8;

    for (int o = 0; o < oc_r4_align; o += 8) {
        auto src_o = src + o * crs;
        auto dst_o = dst + o * crs_r4;
        for (int h = 0; h < kh; h++) {
            for (int w = 0; w < kw; w++) {
                auto src_w = src_o + h * kw + w;
                auto dst_w = dst_o + h * kw * ic_r4 * 8 + w * ic_r4 * 8;
                int c = 0;
                for (; c < ic_align_4; c += 4) {
                    auto src_c = src_w + c * kh * kw;
                    auto dst_c = dst_w + c * 8;
                    for (int i = 0; i < 8; i++) {
                        if (o + i < oc) {
                            for (int j = 0; j < 4; j++) {
                                dst_c[i * 4 + j] = src_c[i * crs + j * kh * kw];
                            }
                        }
                    }
                }
                if (c < ic_r4) {
                    auto src_c = src_w + c * kh * kw;
                    auto dst_c = dst_w + c * 8;
                    for (int i = 0; i < 8; i++) {
                        if (o + i < oc) {
                            for (int j = 0; j < 4; j++) {
                                if (c + j < ic) {
                                    dst_c[i * 4 + j] = src_c[i * crs + j * kh * kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (oc_r4 > oc_r4_align) {
        int o = oc_r4_align;
        auto src_o = src + o * crs;
        auto dst_o = dst + o * crs_r4;
        for (int h = 0; h < kh; h++) {
            for (int w = 0; w < kw; w++) {
                auto src_w = src_o + h * kw + w;
                auto dst_w = dst_o + h * kw * ic_r4 * 4 + w * ic_r4 * 4;
                int c = 0;
                for (; c < ic_align_4; c += 4) {
                    auto src_c = src_w + c * kh * kw;
                    auto dst_c = dst_w + c * 4;
                    for (int i = 0; i < 4; i++) {
                        if (o + i < oc) {
                            for (int j = 0; j < 4; j++) {
                                dst_c[i * 4 + j] = src_c[i * crs + j * kh * kw];
                            }
                        }
                    }
                }
                if (c < ic_r4) {
                    auto src_c = src_w + c * kh * kw;
                    auto dst_c = dst_w + c * 4;
                    for (int i = 0; i < 4; i++) {
                        if (o + i < oc) {
                            for (int j = 0; j < 4; j++) {
                                if (c + j < ic) {
                                    dst_c[i * 4 + j] = src_c[i * crs + j * kh * kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

}   // namespace TNN_NS
