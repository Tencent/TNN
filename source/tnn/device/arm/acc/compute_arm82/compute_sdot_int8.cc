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

namespace TNN_NS {

void PackSDOTINT8Weight(const int8_t *src, int8_t *dst, int oc, int ic, int kh, int kw) {
    int oc_r4  = ROUND_UP(oc, 4);
    int ic_r4  = ROUND_UP(ic, 4);
    int crs    = ic * kw * kh;
    int crs_r4 = ic_r4 * kw * kh;
    int ic_align_4 = ic / 4 * 4;
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

void PackSDOTINT8WeightGemv(const int8_t *src, int8_t *dst, const int oc, const int ic, const int hw) {
    auto ic_r4       = ROUND_UP(ic, 4);
    auto oc_r4       = ROUND_UP(oc, 4);
    auto dst_step    = ic * hw;
    auto dst_step_r4 = ic_r4 * hw;
    int o = 0;
    for (; o + 15 < oc; o += 16) {
        auto dst_o = dst + o * dst_step_r4;
        auto src_o = src + o * dst_step;
        for (int i = 0; i < hw; i++) {
            auto dst_i = dst_o + i * ic_r4 * 16;
            auto src_i = src_o + i;
            int c = 0;
            for (; c + 3 < ic; c += 4) {
                auto dst_c = dst_i + c * 16;
                auto src_c = src_i + c * hw;
                for (int m = 0; m < 16; m++) {
                    for (int n = 0; n < 4; n++) {
                        dst_c[m * 4 + n] = src_c[m * dst_step + n * hw];
                    }
                }
            }
            if (c < ic) {
                auto dst_c = dst_i + c * 16;
                auto src_c = src_i + c * hw;
                for (int m = 0; m < 16; m++) {
                    for (int n = 0; n < ic - c; n++) {
                        dst_c[m * 4 + n] = src_c[m * dst_step + n * hw];
                    }
                }
            }
        }
    }
    for (; o < oc_r4; o += 4) {
        auto dst_o = dst + o * dst_step_r4;
        auto src_o = src + o * dst_step;
        for (int i = 0; i < hw; i++) {
            auto dst_i = dst_o + i * ic_r4 * 4;
            auto src_i = src_o + i;
            int c = 0;
            for (; c + 3 < ic; c += 4) {
                auto dst_c = dst_i + c * 4;
                auto src_c = src_i + c * hw;
                for (int m = 0; m < 4; m++) {
                    if (m + o < oc) {
                        for (int n = 0; n < 4; n++) {
                            dst_c[m * 4 + n] = src_c[m * dst_step + n * hw];
                        }
                    }
                }
            }
            if (c < ic) {
                auto dst_c = dst_i + c * 4;
                auto src_c = src_i + c * hw;
                for (int m = 0; m < 4; m++) {
                    if (m + o < oc) {
                        for (int n = 0; n < ic - c; n++) {
                            dst_c[m * 4 + n] = src_c[m * dst_step + n * hw];
                        }
                    }
                }
            }
        }
    }
}

}   // namespace TNN_NS
