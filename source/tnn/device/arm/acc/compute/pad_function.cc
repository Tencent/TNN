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
#include "pad_function.h"

#include <cstring>

#include "tnn/device/arm/acc/Float4.h"

namespace TNN_NS {

// Common pad in height and width directions
static void CommonPadImpl(float *input_data, float *output_data, int batch_c_r4, int ih, int iw, int oh, int ow,
                          int pad_t, int pad_b, int pad_l, int iw_bytes, Float4 &vvalue) {
    for (int c = 0; c < batch_c_r4; c += 4) {
        auto input_ptr_c  = input_data + c * ih * iw;
        auto output_ptr_c = output_data + c * oh * ow;

        if (pad_t)
            for (int i = 0; i < ow * pad_t; ++i)
                Float4::save(output_ptr_c + i * 4, vvalue);

        for (int h = 0; h < ih; ++h) {
            auto output_ptr_h = output_ptr_c + ow * (h + pad_t) * 4;
            auto input_ptr_h  = input_ptr_c + iw * h * 4;
            for (int i = 0; i < pad_l; i++)
                Float4::save(output_ptr_h + i * 4, vvalue);

            memcpy(output_ptr_h + pad_l * 4, input_ptr_h, iw_bytes);

            for (int i = iw + pad_l; i < ow; i++)
                Float4::save(output_ptr_h + i * 4, vvalue);
        }

        if (pad_b) {
            auto output_ptr_h = output_ptr_c + ow * (ih + pad_t) * 4;
            for (int i = 0; i < ow * pad_b; ++i)
                Float4::save(output_ptr_h + i * 4, vvalue);
        }
    }
}

static void CalculatePad(Float4 &src, const Float4 &vvalue, const int padded_zero) {
    if (padded_zero)
        src = Float4::pad(src, vvalue, padded_zero);
}

// ic_mapped is not alligned to 4 when pad_c_b % 4 != 0
static void ChannelPadNotAligned(float *input_data_base, float *output_ptr_c, int ic_mapped, int ic, int ih, int iw,
                                 int oh, int ow, int pad_t, int pad_b, int pad_l, int pad_r, int iw_bytes,
                                 Float4 &vvalue) {
    int ic_r4 = ROUND_UP(ic, 4);
    // some channel may already be padded with zero
    int padded_zero  = ic_r4 - ic;
    auto ic_mapped_1 = ROUND_UP(ic_mapped, 4);
    auto ic_mapped_0 = ic_mapped_1 - 4;
    // shift_c is used to extract 4 values from two vectors
    auto shift_c = ic_mapped - ic_mapped_0;
    if (ic_mapped_1 < 0 || ic_mapped_0 >= ic_r4) {
        // pad with vvalue
        for (int i = 0; i < ow * oh; ++i)
            Float4::save(output_ptr_c + i * 4, vvalue);
    } else {
        auto input_ptr_c0 = input_data_base + ic_mapped_0 * ih * iw;
        auto input_ptr_c1 = input_data_base + ic_mapped_1 * ih * iw;
        if (pad_t)
            for (int i = 0; i < ow * pad_t; ++i)
                Float4::save(output_ptr_c + i * 4, vvalue);

        for (int h = 0; h < ih; ++h) {
            auto output_ptr_h = output_ptr_c + ow * (h + pad_t) * 4;
            auto input_ptr_h0 = input_ptr_c0 + iw * h * 4;
            auto input_ptr_h1 = input_ptr_c1 + iw * h * 4;
            for (int i = 0; i < pad_l; i++) {
                Float4::save(output_ptr_h, vvalue);
                output_ptr_h += 4;
            }

            if (ic_mapped_0 >= 0 && ic_mapped_1 < ic_r4 - 4) {
                // extract from two vectors
                for (int i = 0; i < iw; i++) {
                    Float4 res = Float4::extract(Float4::load(input_ptr_h0), Float4::load(input_ptr_h1), shift_c);
                    Float4::save(output_ptr_h, res);
                    input_ptr_h0 += 4;
                    input_ptr_h1 += 4;
                    output_ptr_h += 4;
                }
            } else if (ic_mapped_0 < 0) {
                // extract from vvalue && left boundary
                for (int i = 0; i < iw; i++) {
                    Float4 src = Float4::load(input_ptr_h1);
                    if (ic_mapped_1 == ic_r4 - 4)
                        CalculatePad(src, vvalue, padded_zero);
                    Float4 res = Float4::extract(vvalue, src, shift_c);
                    Float4::save(output_ptr_h, res);
                    input_ptr_h1 += 4;
                    output_ptr_h += 4;
                }
            } else if (ic_mapped_1 == ic_r4 - 4) {
                // extract from two vectors, the right one at the boundary
                for (int i = 0; i < iw; i++) {
                    Float4 src = Float4::load(input_ptr_h1);
                    CalculatePad(src, vvalue, padded_zero);
                    Float4 res = Float4::extract(Float4::load(input_ptr_h0), src, shift_c);
                    Float4::save(output_ptr_h, res);
                    input_ptr_h0 += 4;
                    input_ptr_h1 += 4;
                    output_ptr_h += 4;
                }
            } else {
                // extract from right boundary && vvalue
                for (int i = 0; i < iw; i++) {
                    Float4 src = Float4::load(input_ptr_h0);
                    CalculatePad(src, vvalue, padded_zero);
                    Float4 res = Float4::extract(src, vvalue, shift_c);
                    Float4::save(output_ptr_h, res);
                    input_ptr_h0 += 4;
                    output_ptr_h += 4;
                }
            }

            for (int i = 0; i < pad_r; i++) {
                Float4::save(output_ptr_h, vvalue);
                output_ptr_h += 4;
            }
        }

        if (pad_b) {
            auto output_ptr_h = output_ptr_c + ow * (ih + pad_t) * 4;
            for (int i = 0; i < ow * pad_b; ++i)
                Float4::save(output_ptr_h + i * 4, vvalue);
        }
    }
}

// ic_mapped is alligned to 4 when pad_c_b % 4 == 0
static void ChannelPadAligned(float *input_data_base, float *output_ptr_c, int ic_mapped, int ic, int ih, int iw,
                              int oh, int ow, int pad_t, int pad_b, int pad_l, int pad_r, int iw_bytes,
                              Float4 &vvalue) {
    int ic_r4       = ROUND_UP(ic, 4);
    bool ic_aligned = ((ic % 4) == 0);
    // some channel may already be padded with zero
    int padded_zero = ic_r4 - ic;
    if (ic_mapped < 0 || ic_mapped >= ic_r4) {
        for (int i = 0; i < ow * oh; ++i)
            Float4::save(output_ptr_c + i * 4, vvalue);
    } else {
        auto input_ptr_c = input_data_base + ic_mapped * ih * iw;
        if (pad_t)
            for (int i = 0; i < ow * pad_t; ++i)
                Float4::save(output_ptr_c + i * 4, vvalue);

        for (int h = 0; h < ih; ++h) {
            auto output_ptr_h = output_ptr_c + ow * (h + pad_t) * 4;
            auto input_ptr_h  = input_ptr_c + iw * h * 4;
            for (int i = 0; i < pad_l; i++) {
                Float4::save(output_ptr_h, vvalue);
                output_ptr_h += 4;
            }

            if (ic_aligned || ic_mapped <= ic - 4) {
                memcpy(output_ptr_h, input_ptr_h, iw_bytes);
                output_ptr_h += iw * 4;
            } else {
                for (int i = 0; i < iw; i++) {
                    Float4 res = Float4::pad(Float4::load(input_ptr_h), vvalue, padded_zero);
                    Float4::save(output_ptr_h, res);
                    input_ptr_h += 4;
                    output_ptr_h += 4;
                }
            }

            for (int i = 0; i < pad_r; i++) {
                Float4::save(output_ptr_h, vvalue);
                output_ptr_h += 4;
            }
        }
        if (pad_b) {
            auto output_ptr_h = output_ptr_c + ow * (ih + pad_t) * 4;
            for (int i = 0; i < ow * pad_b; ++i)
                Float4::save(output_ptr_h + i * 4, vvalue);
        }
    }
}

// Channel pad in channel, height and width directions
static void ChannelPadImpl(float *input_data, float *output_data, int batch, int c_r4, int oh, int ow, int ic, int ih,
                           int iw, int pad_t, int pad_b, int pad_l, int pad_r, int pad_c_b, int pad_c_e, int iw_bytes,
                           Float4 &vvalue) {
    int ic_r4          = ROUND_UP(ic, 4);
    bool pad_c_aligned = ((pad_c_b % 4) == 0);
    for (int n = 0; n < batch; ++n) {
        auto input_data_base  = input_data + n * ic_r4 * ih * iw;
        auto output_data_base = output_data + n * c_r4 * oh * ow;
        for (int c = 0; c < c_r4; c += 4) {
            auto output_ptr_c = output_data_base + c * oh * ow;
            auto ic_mapped    = c - pad_c_b;
            if (pad_c_aligned) {
                ChannelPadAligned(input_data_base, output_ptr_c, ic_mapped, ic, ih, iw, oh, ow, pad_t, pad_b, pad_l,
                                  pad_r, iw_bytes, vvalue);
            } else {
                ChannelPadNotAligned(input_data_base, output_ptr_c, ic_mapped, ic, ih, iw, oh, ow, pad_t, pad_b, pad_l,
                                     pad_r, iw_bytes, vvalue);
            }
        }
    }
}

Status PadUtils::ConstPadV2(float *input_data, float *output_data, DimsVector input_dims, DimsVector output_dims,
                            PadContext context) {
    if (input_dims.size() < 2 || input_dims.size() > 5) {
        LOGE("Arm PadV2(const type) only support 2 - 5 dims\n");
        return Status(TNNERR_UNKNOWN_LAYER, "Arm PadV2 only support 2 - 5 dims");
    }
    const int batch    = context.output_batch;
    const int oc_r4    = context.output_channel_r4;
    const int oh       = context.output_height;
    const int ow       = context.output_width;
    const int ic       = context.input_channel;
    const int ih       = context.input_height;
    const int iw       = context.input_width;
    const int iw_bytes = iw * sizeof(float) * 4;
    const int pad_c_b  = context.pad_c_b;
    const int pad_c_e  = context.pad_c_e;
    const int pad_t    = context.pad_t;
    const int pad_l    = context.pad_l;
    const int pad_b    = context.pad_b;
    const int pad_r    = context.pad_r;
    Float4 value_v     = Float4(context.value);
    
    //ncdhw, extend dim except the batch n
    if (context.input_batch == context.output_batch) {
    if (pad_c_b == 0 && pad_c_e == 0) {
        CommonPadImpl(input_data, output_data, batch * oc_r4, ih, iw, oh, ow, pad_t, pad_b, pad_l, iw_bytes, value_v);
    } else {
        ChannelPadImpl(input_data, output_data, batch, oc_r4, oh, ow, ic, ih, iw, pad_t, pad_b, pad_l, pad_r, pad_c_b,
                       pad_c_e, iw_bytes, value_v);
    }
    } else {
        //ncdhw, only extend the batch n
        if (context.input_channel == context.output_channel &&
            context.input_depth == context.output_depth &&
            context.input_height == context.output_height &&
            context.input_width == context.output_width) {
            const int batch_size = context.input_channel_r4*context.input_depth*context.input_height*context.input_width;
            //batch begin
            for (int i=0; i<context.pad_b_b*batch_size/4; i++) {
                Float4::save(output_data += 4, value_v);
            }
            //input data
            const int input_size = context.input_batch * batch_size;
            memcpy(output_data , input_data, input_size * sizeof(float));
            output_data += input_size;
            
            //batch end
            for (int i=0; i<context.pad_b_e*batch_size/4; i++) {
                Float4::save(output_data += 4, value_v);
            }
        } else {
            LOGE("Arm PadV2(const type) dont support pad with batch and other dim at the same time\n");
            return Status(TNNERR_UNKNOWN_LAYER, "Arm PadV2(const type) dont support pad with batch and other dim at the same time");
        }
    }
    return TNN_OK;
}

Status PadUtils::ReflectPadV2(float *input_data, float *output_data, DimsVector input_dims, DimsVector output_dims,
                              PadContext context) {
    if (input_dims.size() < 2 || input_dims.size() > 5) {
        LOGE("Arm PadV2(reflect type) only support 2 - 5 dims\n");
        return Status(TNNERR_UNKNOWN_LAYER, "Arm PadV2 only support 2 - 5 dims");
    }
    
    const int batch     = context.output_batch;
    const int c_r4      = context.output_channel_r4;
    const int oh        = context.output_height;
    const int ow        = context.output_width;
    const int ic        = context.input_channel;
    const int ih        = context.input_height;
    const int iw        = context.input_width;
    const int byte_size = sizeof(float);
    const int iw_bytes  = iw * byte_size * 4;
    const int pad_c_b   = context.pad_c_b;
    const int pad_c_e   = context.pad_c_e;
    const int pad_t     = context.pad_t;
    const int pad_l     = context.pad_l;
    const int pad_b     = context.pad_b;
    const int pad_r     = context.pad_r;
    
    //ncdhw, extend dim except the batch n
    if (context.input_batch == context.output_batch) {
    for (int c = 0; c < batch * c_r4; c += 4) {
        auto input_ptr_c  = input_data + c * ih * iw;
        auto output_ptr_c = output_data + c * oh * ow;

        for (int h = 0; h < ih; ++h) {
            auto output_ptr_h = output_ptr_c + ow * (h + pad_t) * 4;
            auto input_ptr_h  = input_ptr_c + iw * h * 4;
            for (int i = 0; i < pad_l; i++) {
                Float4::save(output_ptr_h + i * 4, Float4::load(input_ptr_h + (pad_l - i) * 4));
            }

            memcpy(output_ptr_h + pad_l * 4, input_ptr_h, iw_bytes);

            for (int i = 0; i < pad_r; i++) {
                Float4::save(output_ptr_h + (i + pad_l + iw) * 4, Float4::load(input_ptr_h + (iw - 1 - (i + 1)) * 4));
            }
        }
        // pad: copy from output
        for (int h = 0; h < pad_t; h++) {
            auto output_ptr_h = output_ptr_c + ow * h * 4;
            auto output_ref_h = output_ptr_c + ow * (pad_t + pad_t - h) * 4;
            memcpy(output_ptr_h, output_ref_h, ow * byte_size * 4);
        }

        for (int h = 0; h < pad_b; h++) {
            auto output_ptr_h = output_ptr_c + ow * (h + ih + pad_t) * 4;
            auto output_ref_h = output_ptr_c + ow * (ih + pad_t - 1 - (h + 1)) * 4;
            memcpy(output_ptr_h, output_ref_h, ow * byte_size * 4);
        }
    }
    } else {
        LOGE("Arm PadV2(reflect type) dont support pad with batch and other dim at the same time\n");
        return Status(TNNERR_UNKNOWN_LAYER, "Arm PadV2(reflect type) dont support pad with batch and other dim at the same time");
    }
    return TNN_OK;
}

}  // namespace TNN_NS
