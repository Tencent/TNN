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

#include "tnn/device/x86/acc/convolution/x86_conv_layer_3x3.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/device/x86/x86_common.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/device/x86/x86_util.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

static void weight_transform(const float *src, float *dst, int kernel_size, int unit, int in_channel, int out_channel,
                             const float (*G)[3]) {
    float M[unit][3];
    float K_trans[unit * unit];

    int ic_8        = UP_DIV(in_channel, 8);
    int oc_8        = UP_DIV(out_channel, 8);
    int unit_stride = ic_8 * oc_8 * 8 * 8;
    int oc_stride   = ic_8 * 8 * 8;
    int ic_stride   = 8 * 8;

    for (int oc = 0; oc < out_channel; oc++) {
        int zo        = oc / 8;
        int ro        = oc % 8;
        float *dst_oz = dst + zo * oc_stride + ro;
        for (int ic = 0; ic < in_channel; ic++) {
            const float *src_z = src + (oc * in_channel + ic) * 3 * 3;
            const float *k0    = src_z;
            const float *k1    = k0 + 3;
            const float *k2    = k1 + 3;

            int zi = ic / 8;
            int ri = ic % 8;

            // M=G*g
            for (int i = 0; i < unit; i++) {
                M[i][0] = k0[0] * G[i][0] + k1[0] * G[i][1] + k2[0] * G[i][2];
                M[i][1] = k0[1] * G[i][0] + k1[1] * G[i][1] + k2[1] * G[i][2];
                M[i][2] = k0[2] * G[i][0] + k1[2] * G[i][1] + k2[2] * G[i][2];
            }

            // K_trans=M*GT
            for (int j = 0; j < unit; j++) {
                float *Mp = &M[j][0];
                for (int i = 0; i < unit; i++) {
                    K_trans[j * unit + i] = Mp[0] * G[i][0] + Mp[1] * G[i][1] + Mp[2] * G[i][2];
                }
            }

            auto dst_sz = dst_oz + zi * ic_stride + 8 * ri;

            for (int i = 0; i < unit; i++) {
                for (int j = 0; j < unit; j++) {
                    *(dst_sz + (j * unit + i) * unit_stride) = K_trans[i * unit + j];
                }
            }
        }
    }
}

static void pack_input_c8(const float *din, float *dout, int cs, int hs, int he, int ws, int we, int channel, int width,
                          int height, float *zero_ptr) {
    int size_w = we - ws;
    int size_c = width * height;
    int pad_l  = ws < 0 ? -ws : 0;
    int pad_r  = we > width ? we - width : 0;
    auto dst   = dout;

    for (int h = hs; h < he; h++) {
        dst         = dout + (h - hs) * 8 * size_w;
        auto ptr_c0 = din + cs * size_c + h * width;
        auto ptr_c1 = ptr_c0 + size_c;
        auto ptr_c2 = ptr_c1 + size_c;
        auto ptr_c3 = ptr_c2 + size_c;
        auto ptr_c4 = ptr_c3 + size_c;
        auto ptr_c5 = ptr_c4 + size_c;
        auto ptr_c6 = ptr_c5 + size_c;
        auto ptr_c7 = ptr_c6 + size_c;

        if (h < 0 || h >= height) {
            memset(dst, 0, sizeof(float) * 8 * size_w);
            continue;
        } else if (cs + 8 > channel) {
            switch (cs + 8 - channel) {
                case 7:
                    ptr_c1 = zero_ptr;
                case 6:
                    ptr_c2 = zero_ptr;
                case 5:
                    ptr_c3 = zero_ptr;
                case 4:
                    ptr_c4 = zero_ptr;
                case 3:
                    ptr_c5 = zero_ptr;
                case 2:
                    ptr_c6 = zero_ptr;
                case 1:
                    ptr_c7 = zero_ptr;
                default:
                    break;
            }
        }

        for (int w = ws; w < we; w++) {
            if (w < 0 || w >= width) {
                memset(dst, 0, 8 * sizeof(float));
                dst += 8;
                continue;
            }

            dst[0] = ptr_c0[w];
            dst[1] = ptr_c1[w];
            dst[2] = ptr_c2[w];
            dst[3] = ptr_c3[w];
            dst[4] = ptr_c4[w];
            dst[5] = ptr_c5[w];
            dst[6] = ptr_c6[w];
            dst[7] = ptr_c7[w];
            dst += 8;
        }
    }
}

// unpack c8
static void unpack_output_c8(const float *din, float *dout, int cs, int ce, int hs, int he, int ws, int we, int channel,
                             int height, int width, bool flag_relu, float *trash_ptr) {
    int size_c_out = width * height;

    float *doutc0r0 = dout + cs * size_c_out + hs * width + ws;
    float *doutc1r0 = doutc0r0 + size_c_out;
    float *doutc2r0 = doutc1r0 + size_c_out;
    float *doutc3r0 = doutc2r0 + size_c_out;
    float *doutc4r0 = doutc3r0 + size_c_out;
    float *doutc5r0 = doutc4r0 + size_c_out;
    float *doutc6r0 = doutc5r0 + size_c_out;
    float *doutc7r0 = doutc6r0 + size_c_out;

    const float *ptr_din = din;

    int size_h  = (he > height ? height : he) - hs;
    int size_w  = (we > width ? width : we) - ws;
    int valid_w = we - ws;

    for (int h = 0; h < size_h; h++) {
        float *doutc0_ptr = doutc0r0 + h * width;  // doutc0r0 + width;
        float *doutc1_ptr = doutc1r0 + h * width;
        float *doutc2_ptr = doutc2r0 + h * width;
        float *doutc3_ptr = doutc3r0 + h * width;
        float *doutc4_ptr = doutc4r0 + h * width;
        float *doutc5_ptr = doutc5r0 + h * width;
        float *doutc6_ptr = doutc6r0 + h * width;
        float *doutc7_ptr = doutc7r0 + h * width;
        if (ce > channel) {
            switch (ce - channel) {
                case 7:
                    doutc1_ptr = trash_ptr;
                case 6:
                    doutc2_ptr = trash_ptr;
                case 5:
                    doutc3_ptr = trash_ptr;
                case 4:
                    doutc4_ptr = trash_ptr;
                case 3:
                    doutc5_ptr = trash_ptr;
                case 2:
                    doutc6_ptr = trash_ptr;
                case 1:
                    doutc7_ptr = trash_ptr;
                default:
                    break;
            }
        }
        for (int w = 0; w < size_w; w++) {
            doutc0_ptr[w] = ptr_din[(h * valid_w + w) * 8 + 0];
            doutc1_ptr[w] = ptr_din[(h * valid_w + w) * 8 + 1];
            doutc2_ptr[w] = ptr_din[(h * valid_w + w) * 8 + 2];
            doutc3_ptr[w] = ptr_din[(h * valid_w + w) * 8 + 3];
            doutc4_ptr[w] = ptr_din[(h * valid_w + w) * 8 + 4];
            doutc5_ptr[w] = ptr_din[(h * valid_w + w) * 8 + 5];
            doutc6_ptr[w] = ptr_din[(h * valid_w + w) * 8 + 6];
            doutc7_ptr[w] = ptr_din[(h * valid_w + w) * 8 + 7];
        }
    }
}

#define COMPUTE_UNIT(c)                                                                                                \
    wgt  = _mm256_loadu_ps(weight_z + c * N);                                                                          \
    data = _mm256_broadcast_ss(src_z + K * 0 + c);                                                                     \
    acc0 = _mm256_fmadd_ps(data, wgt, acc0);                                                                           \
    data = _mm256_broadcast_ss(src_z + K * 1 + c);                                                                     \
    acc1 = _mm256_fmadd_ps(data, wgt, acc1);                                                                           \
    data = _mm256_broadcast_ss(src_z + K * 2 + c);                                                                     \
    acc2 = _mm256_fmadd_ps(data, wgt, acc2);                                                                           \
    data = _mm256_broadcast_ss(src_z + K * 3 + c);                                                                     \
    acc3 = _mm256_fmadd_ps(data, wgt, acc3);                                                                           \
    data = _mm256_broadcast_ss(src_z + K * 4 + c);                                                                     \
    acc4 = _mm256_fmadd_ps(data, wgt, acc4);                                                                           \
    data = _mm256_broadcast_ss(src_z + K * 5 + c);                                                                     \
    acc5 = _mm256_fmadd_ps(data, wgt, acc5);

#define COMPUTE_VEC(c)                                                                                                 \
    data = _mm256_broadcast_ss(src_z + c);                                                                             \
    wgt  = _mm256_loadu_ps(weight_z + c * N);                                                                          \
    acc  = _mm256_fmadd_ps(data, wgt, acc);

// A=6x8, B=8x8, C=6x8
void gemm_kernel(float *dst, float *src, const float *weight, const float *bias, int ic_8, int oc_8, int width) {
    const size_t M = 6;
    const size_t K = 8;
    const size_t N = 8;

    auto w_unit         = width / M;
    auto w_unit_end     = M * w_unit;
    auto src_depth_step = width * K;

    const size_t weight_unit_step = K * N;

    for (int co = 0; co < oc_8; co++) {
        auto dst_z     = dst + co * N * width;
        auto weight_dz = weight + co * ic_8 * weight_unit_step;

        for (int dx = 0; dx < w_unit; dx++) {
            auto dst_x = dst_z + dx * N * M;
            auto src_x = src + dx * K * M;

            register __m256 acc0 asm("ymm0") = (bias) ? _mm256_loadu_ps(bias) : _mm256_set1_ps(0.0f);
            register __m256 acc1 asm("ymm1") = acc0;
            register __m256 acc2 asm("ymm2") = acc0;
            register __m256 acc3 asm("ymm3") = acc0;
            register __m256 acc4 asm("ymm4") = acc0;
            register __m256 acc5 asm("ymm5") = acc0;
            register __m256 data asm("ymm6");
            register __m256 wgt asm("ymm7");

            for (int ci = 0; ci < ic_8; ci++) {
                auto src_z    = src_x + ci * src_depth_step;
                auto weight_z = weight_dz + ci * weight_unit_step;

                COMPUTE_UNIT(0);
                COMPUTE_UNIT(1);
                COMPUTE_UNIT(2);
                COMPUTE_UNIT(3);
                COMPUTE_UNIT(4);
                COMPUTE_UNIT(5);
                COMPUTE_UNIT(6);
                COMPUTE_UNIT(7);
            }

            _mm256_storeu_ps(dst_x + N * 0, acc0);
            _mm256_storeu_ps(dst_x + N * 1, acc1);
            _mm256_storeu_ps(dst_x + N * 2, acc2);
            _mm256_storeu_ps(dst_x + N * 3, acc3);
            _mm256_storeu_ps(dst_x + N * 4, acc4);
            _mm256_storeu_ps(dst_x + N * 5, acc5);
        }

        for (int dx = w_unit_end; dx < width; dx++) {
            auto dst_x = dst_z + dx * N;
            auto src_x = src + dx * K;

            register __m256 acc asm("ymm0") = (bias) ? _mm256_loadu_ps(bias) : _mm256_set1_ps(0.0f);
            register __m256 data asm("ymm6");
            register __m256 wgt asm("ymm7");

            for (int ci = 0; ci < ic_8; ci++) {
                auto src_z    = src_x + ci * src_depth_step;
                auto weight_z = weight_dz + ci * weight_unit_step;

                COMPUTE_VEC(0);
                COMPUTE_VEC(1);
                COMPUTE_VEC(2);
                COMPUTE_VEC(3);
                COMPUTE_VEC(4);
                COMPUTE_VEC(5);
                COMPUTE_VEC(6);
                COMPUTE_VEC(7);
            }

            _mm256_storeu_ps(dst_x, acc);
        }
    }
}

// BT=[1, 0, -1, 0,
//    0, 1,  1, 0,
//    0, -1, 1, 0,
//    0, 1,  0, -1]
void input_trans_c8_4x4(const float *src, int src_stride, int src_h_stride, float *dest, int dest_stride,
                        int dest_h_stride) {
    __m256 src00 = _mm256_loadu_ps(src);
    __m256 src01 = _mm256_loadu_ps(src + src_stride);
    __m256 src02 = _mm256_loadu_ps(src + src_stride + src_stride);
    __m256 src03 = _mm256_loadu_ps(src + src_stride + src_stride + src_stride);
    src += src_h_stride;
    __m256 src10 = _mm256_loadu_ps(src);
    __m256 src11 = _mm256_loadu_ps(src + src_stride);
    __m256 src12 = _mm256_loadu_ps(src + src_stride + src_stride);
    __m256 src13 = _mm256_loadu_ps(src + src_stride + src_stride + src_stride);
    src += src_h_stride;
    __m256 src20 = _mm256_loadu_ps(src);
    __m256 src21 = _mm256_loadu_ps(src + src_stride);
    __m256 src22 = _mm256_loadu_ps(src + src_stride + src_stride);
    __m256 src23 = _mm256_loadu_ps(src + src_stride + src_stride + src_stride);
    src += src_h_stride;
    __m256 src30 = _mm256_loadu_ps(src);
    __m256 src31 = _mm256_loadu_ps(src + src_stride);
    __m256 src32 = _mm256_loadu_ps(src + src_stride + src_stride);
    __m256 src33 = _mm256_loadu_ps(src + src_stride + src_stride + src_stride);

    __m256 dst00 = _mm256_sub_ps(src00, src02);
    __m256 dst10 = _mm256_add_ps(src01, src02);
    __m256 dst20 = _mm256_sub_ps(src02, src01);
    __m256 dst30 = _mm256_sub_ps(src01, src03);

    __m256 dst01 = _mm256_sub_ps(src10, src12);
    __m256 dst11 = _mm256_add_ps(src11, src12);
    __m256 dst21 = _mm256_sub_ps(src12, src11);
    __m256 dst31 = _mm256_sub_ps(src11, src13);

    __m256 dst02 = _mm256_sub_ps(src20, src22);
    __m256 dst12 = _mm256_add_ps(src21, src22);
    __m256 dst22 = _mm256_sub_ps(src22, src21);
    __m256 dst32 = _mm256_sub_ps(src21, src23);

    __m256 dst03 = _mm256_sub_ps(src30, src32);
    __m256 dst13 = _mm256_add_ps(src31, src32);
    __m256 dst23 = _mm256_sub_ps(src32, src31);
    __m256 dst33 = _mm256_sub_ps(src31, src33);

    __m256 dest00 = _mm256_sub_ps(dst00, dst02);
    __m256 dest10 = _mm256_add_ps(dst01, dst02);
    __m256 dest20 = _mm256_sub_ps(dst02, dst01);
    __m256 dest30 = _mm256_sub_ps(dst01, dst03);

    __m256 dest01 = _mm256_sub_ps(dst10, dst12);
    __m256 dest11 = _mm256_add_ps(dst11, dst12);
    __m256 dest21 = _mm256_sub_ps(dst12, dst11);
    __m256 dest31 = _mm256_sub_ps(dst11, dst13);

    __m256 dest02 = _mm256_sub_ps(dst20, dst22);
    __m256 dest12 = _mm256_add_ps(dst21, dst22);
    __m256 dest22 = _mm256_sub_ps(dst22, dst21);
    __m256 dest32 = _mm256_sub_ps(dst21, dst23);

    __m256 dest03 = _mm256_sub_ps(dst30, dst32);
    __m256 dest13 = _mm256_add_ps(dst31, dst32);
    __m256 dest23 = _mm256_sub_ps(dst32, dst31);
    __m256 dest33 = _mm256_sub_ps(dst31, dst33);

    _mm256_storeu_ps(dest, dest00);
    _mm256_storeu_ps(dest + dest_stride, dest10);
    _mm256_storeu_ps(dest + dest_stride + dest_stride, dest20);
    _mm256_storeu_ps(dest + dest_stride + dest_stride + dest_stride, dest30);
    dest += dest_h_stride;
    _mm256_storeu_ps(dest, dest01);
    _mm256_storeu_ps(dest + dest_stride, dest11);
    _mm256_storeu_ps(dest + dest_stride + dest_stride, dest21);
    _mm256_storeu_ps(dest + dest_stride + dest_stride + dest_stride, dest31);
    dest += dest_h_stride;
    _mm256_storeu_ps(dest, dest02);
    _mm256_storeu_ps(dest + dest_stride, dest12);
    _mm256_storeu_ps(dest + dest_stride + dest_stride, dest22);
    _mm256_storeu_ps(dest + dest_stride + dest_stride + dest_stride, dest32);
    dest += dest_h_stride;
    _mm256_storeu_ps(dest, dest03);
    _mm256_storeu_ps(dest + dest_stride, dest13);
    _mm256_storeu_ps(dest + dest_stride + dest_stride, dest23);
    _mm256_storeu_ps(dest + dest_stride + dest_stride + dest_stride, dest33);
}

// AT=[1, 1,  1,  0,
//    0, 1, -1, -1
void output_trans_c8_post_2x4(const float *src, int src_stride, int src_h_stride, float *dest, int dest_stride,
                              int dest_h_stride, float *bias_value, bool has_relu) {
    __m256 src00 = _mm256_loadu_ps(src);
    __m256 src01 = _mm256_loadu_ps(src + src_stride);
    __m256 src02 = _mm256_loadu_ps(src + src_stride + src_stride);
    __m256 src03 = _mm256_loadu_ps(src + src_stride + src_stride + src_stride);
    src += src_h_stride;
    __m256 src10 = _mm256_loadu_ps(src);
    __m256 src11 = _mm256_loadu_ps(src + src_stride);
    __m256 src12 = _mm256_loadu_ps(src + src_stride + src_stride);
    __m256 src13 = _mm256_loadu_ps(src + src_stride + src_stride + src_stride);
    src += src_h_stride;
    __m256 src20 = _mm256_loadu_ps(src);
    __m256 src21 = _mm256_loadu_ps(src + src_stride);
    __m256 src22 = _mm256_loadu_ps(src + src_stride + src_stride);
    __m256 src23 = _mm256_loadu_ps(src + src_stride + src_stride + src_stride);
    src += src_h_stride;
    __m256 src30 = _mm256_loadu_ps(src);
    __m256 src31 = _mm256_loadu_ps(src + src_stride);
    __m256 src32 = _mm256_loadu_ps(src + src_stride + src_stride);
    __m256 src33 = _mm256_loadu_ps(src + src_stride + src_stride + src_stride);

    __m256 dst00 = _mm256_add_ps(_mm256_add_ps(src00, src01), src02);
    __m256 dst10 = _mm256_sub_ps(_mm256_sub_ps(src01, src02), src03);
    __m256 dst01 = _mm256_add_ps(_mm256_add_ps(src10, src11), src12);
    __m256 dst11 = _mm256_sub_ps(_mm256_sub_ps(src11, src12), src13);
    __m256 dst02 = _mm256_add_ps(_mm256_add_ps(src20, src21), src22);
    __m256 dst12 = _mm256_sub_ps(_mm256_sub_ps(src21, src22), src23);
    __m256 dst03 = _mm256_add_ps(_mm256_add_ps(src30, src31), src32);
    __m256 dst13 = _mm256_sub_ps(_mm256_sub_ps(src31, src32), src33);

    __m256 dest00 = _mm256_add_ps(_mm256_add_ps(dst00, dst01), dst02);
    __m256 dest10 = _mm256_sub_ps(_mm256_sub_ps(dst01, dst02), dst03);
    __m256 dest01 = _mm256_add_ps(_mm256_add_ps(dst10, dst11), dst12);
    __m256 dest11 = _mm256_sub_ps(_mm256_sub_ps(dst11, dst12), dst13);

    if (bias_value) {
        __m256 bias = _mm256_loadu_ps(bias_value);
        dest00      = _mm256_add_ps(dest00, bias);
        dest10      = _mm256_add_ps(dest10, bias);
        dest01      = _mm256_add_ps(dest01, bias);
        dest11      = _mm256_add_ps(dest11, bias);
    }

    if (has_relu) {
        __m256 zeros = _mm256_set1_ps(0);
        dest00       = _mm256_max_ps(dest00, zeros);
        dest10       = _mm256_max_ps(dest10, zeros);
        dest01       = _mm256_max_ps(dest01, zeros);
        dest11       = _mm256_max_ps(dest11, zeros);
    }

    _mm256_storeu_ps(dest, dest00);
    _mm256_storeu_ps(dest + dest_stride, dest10);
    dest += dest_h_stride;
    _mm256_storeu_ps(dest, dest01);
    _mm256_storeu_ps(dest + dest_stride, dest11);
}

bool X86ConvLayer3x3::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    if (param->group != 1) {
        return false;
    }

    const int kw = param->kernels[0];
    const int kh = param->kernels[1];
    const int dw = param->dialations[0];
    const int dh = param->dialations[1];
    const int sw = param->strides[0];
    const int sh = param->strides[1];
    const int ic = inputs[0]->GetBlobDesc().dims[1];

    return kw == 3 && kh == 3 && dw == 1 && dh == 1 && sw == 1 && sh == 1 && ic >= 16;
}

X86ConvLayer3x3::~X86ConvLayer3x3() {}

Status X86ConvLayer3x3::allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    if (!buffer_weight_.GetBytesSize()) {
        const float *src = conv_res->filter_handle.force_to<float *>();

        const int input_channel  = dims_input[1];
        const int output_channel = dims_output[1];
        const int weight_count   = ROUND_UP(input_channel, 8) * ROUND_UP(output_channel, 8) * 4 * 4;
        const int data_byte_size = DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());

        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
            RawBuffer pack_buffer(weight_count * data_byte_size);
            float *dst = pack_buffer.force_to<float *>();

            const float G[4][3] = {{1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}};
            weight_transform(src, dst, 3, 4, input_channel, output_channel, G);

            pack_buffer.SetDataType(DATA_TYPE_FLOAT);
            buffer_weight_ = pack_buffer;
        } else {
            LOGE("Error: DataType %d not support\n", conv_res->filter_handle.GetDataType());
            return Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        }
    }
    return TNN_OK;
}

// pack weight offline
// pack input c8
// input trans
// gemm
// output trans
// write c8 to nchw

#define TILE_NUM 6
#define CH_PACK 8

Status X86ConvLayer3x3::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    float *weights_data = buffer_weight_.force_to<float *>();
    float *bias_data    = buffer_bias_.force_to<float *>();

    auto src_origin = reinterpret_cast<float *>(input->GetHandle().base);
    auto dst_origin = reinterpret_cast<float *>(output->GetHandle().base);

    const int batch       = dims_output[0];
    const int channel_in  = dims_input[1];
    const int height_in   = dims_input[2];
    const int width_in    = dims_input[3];
    const int channel_out = dims_output[1];
    const int height_out  = dims_output[2];
    const int width_out   = dims_output[3];

    const int pad_left   = param->pads[0];
    const int pad_right  = param->pads[1];
    const int pad_top    = param->pads[2];
    const int pad_bottom = param->pads[3];

    size_t in_n_stride  = channel_in * width_in * height_in;
    size_t out_n_stride = channel_out * width_out * height_out;
    size_t ic_stride    = width_in * height_in;
    size_t oc_stride    = width_out * height_out;

    size_t ic_8 = UP_DIV(channel_in, CH_PACK);
    size_t oc_8 = UP_DIV(channel_out, CH_PACK);

    const size_t src_unit = 4;
    const size_t dst_unit = 2;
    size_t w_unit         = UP_DIV(width_out, dst_unit);
    size_t h_unit         = UP_DIV(height_out, dst_unit);
    size_t total_cnt      = UP_DIV(w_unit * h_unit, TILE_NUM);

    size_t w_pad = width_in + pad_left + pad_right;
    size_t h_pad = height_in + pad_top + pad_bottom;

    const size_t zero_len = w_pad;
    float zero_ptr[zero_len];
    memset(zero_ptr, 0, sizeof(float) * zero_len);
    float *pack_input = (float *)_mm_malloc(w_pad * h_pad * ROUND_UP(channel_in, CH_PACK) * sizeof(float), 32);
    float *input_c8   = pack_input;

    size_t new_h_stride = w_pad * CH_PACK;
    size_t new_c_stride = new_h_stride * h_pad;
    size_t ic_8_stride  = w_pad * h_pad * CH_PACK;
    size_t oc_8_stride  = width_out * height_out * CH_PACK;

    float *tmp_data = (float *)_mm_malloc((ic_8 + oc_8) * src_unit * src_unit * CH_PACK * TILE_NUM * sizeof(float), 32);
    float *src_trans_tmp_data = (float *)_mm_malloc(src_unit * src_unit * CH_PACK * sizeof(float), 32);
    float *dst_trans_tmp_data = (float *)_mm_malloc(dst_unit * dst_unit * CH_PACK * sizeof(float), 32);

    for (int ni = 0; ni < batch; ni++) {
        auto input_ptr  = src_origin + ni * in_n_stride;
        auto output_ptr = dst_origin + ni * out_n_stride;

        for (int i = 0; i < ic_8; ++i) {
            pack_input_c8(input_ptr, input_c8 + i * new_c_stride, i * CH_PACK, -pad_top, height_in + pad_bottom,
                          -pad_left, width_in + pad_right, channel_in, width_in, height_in, zero_ptr);
        }
        const float *weight_ptr = buffer_weight_.force_to<float *>();
        const float *bias_ptr   = buffer_bias_.force_to<float *>();

        for (int t_idx = 0; t_idx < total_cnt; t_idx++) {
            int tile_index  = t_idx * TILE_NUM;
            int tile_remain = w_unit * h_unit - tile_index;
            int tile_count  = tile_remain > TILE_NUM ? TILE_NUM : tile_remain;

            // ----------------------------------------- input trans -------------------------------------
            int c_gi_stride = tile_count * oc_8 * CH_PACK;
            int b_gi_stride = tile_count * ic_8 * CH_PACK;

            for (int x_i = 0; x_i < tile_count; x_i++) {
                int index = tile_index + x_i;
                int w_idx = index % w_unit;
                int h_idx = index / w_unit;

                int src_x = w_idx * dst_unit;
                int src_y = h_idx * dst_unit;
                int ex    = src_x + src_unit > w_pad ? w_pad - src_x : src_unit;
                int ey    = src_y + src_unit > h_pad ? h_pad - src_y : src_unit;

                float *dst_ptr       = tmp_data + x_i * CH_PACK;
                const float *src_ptr = input_c8 + (src_y * w_pad + src_x) * CH_PACK;

                if (ex == src_unit && ey == src_unit) {
                    // trans input
                    for (int ci = 0; ci < ic_8; ++ci) {
                        const float *src_ci = src_ptr + ci * ic_8_stride;
                        float *dst_ci       = dst_ptr + ci * tile_count * CH_PACK;
                        input_trans_c8_4x4(src_ci, CH_PACK, w_pad * CH_PACK, dst_ci, b_gi_stride,
                                           b_gi_stride * src_unit);
                    }
                } else {
                    int x_size = ex;
                    for (int ci = 0; ci < ic_8; ++ci) {
                        const float *src_ci = src_ptr + ci * ic_8_stride;
                        // pad
                        memset(src_trans_tmp_data, 0, 128 * sizeof(float));  // src_unit * src_unit * ch_pack
                        if (x_size > 0) {
                            for (int yi = 0; yi < ey; ++yi) {
                                float *dst_yi       = src_trans_tmp_data + yi * src_unit * CH_PACK;
                                const float *src_yi = src_ci + w_pad * yi * CH_PACK;
                                memcpy(dst_yi, src_yi, x_size * sizeof(float) * CH_PACK);
                            }
                        }

                        // trans
                        float *dst_ci = dst_ptr + ci * tile_count * CH_PACK;
                        input_trans_c8_4x4(src_trans_tmp_data, CH_PACK, src_unit * CH_PACK, dst_ci, b_gi_stride,
                                           b_gi_stride * src_unit);
                    }
                }
            }

            // ---------------------------------------- gemm func ----------------------------------------
            // gemm
            float *dst_temp_data = tmp_data + TILE_NUM * ic_8 * 128;  // src_unit * src_unit * ch_pack
            float *b_ptr         = tmp_data;
            int w_gi_stride      = ic_8 * oc_8 * CH_PACK * CH_PACK;
            for (int gi = 0; gi < src_unit * src_unit; ++gi) {
                float *trans_dst          = dst_temp_data + gi * c_gi_stride;
                float *trans_src          = b_ptr + gi * b_gi_stride;
                const float *trans_weight = weight_ptr + gi * w_gi_stride;

                gemm_kernel(trans_dst, trans_src, trans_weight, nullptr, ic_8, oc_8, tile_count);
            }

            // ---------------------------------------- output trans --------------------------------------
            float bias_value[CH_PACK];
            memset(bias_value, 0, CH_PACK * sizeof(float));

            for (int ti = 0; ti < tile_count; ++ti) {
                int index = tile_index + ti;

                int w_idx = index % w_unit;
                int h_idx = index / w_unit;

                int dst_x = w_idx * dst_unit;
                int dst_y = h_idx * dst_unit;

                int ex = dst_x + dst_unit > width_out ? width_out - dst_x : dst_unit;
                int ey = dst_y + dst_unit > height_out ? height_out - dst_y : dst_unit;

                float *dst_ptr = output_ptr + (dst_y * width_out + dst_x) * CH_PACK;
                float *src_ptr = dst_temp_data + ti * CH_PACK;

                if (ex == 2) {
                    // trans output
                    for (int ci = 0; ci < oc_8; ++ci) {
                        if (bias_ptr) {
                            memcpy(bias_value, bias_ptr + ci * CH_PACK, CH_PACK * sizeof(float));
                        }

                        float *dst_ci = dst_ptr + ci * oc_8_stride;
                        float *src_ci = src_ptr + ci * tile_count * CH_PACK;
                        output_trans_c8_post_2x4(src_ci, c_gi_stride, c_gi_stride * src_unit, src_trans_tmp_data,
                                                 CH_PACK, dst_unit * CH_PACK, bias_value, param->activation_type);
                        unpack_output_c8(src_trans_tmp_data, output_ptr, ci * CH_PACK, ci * CH_PACK + CH_PACK, dst_y,
                                         dst_y + ey, dst_x, dst_x + ex, channel_out, height_out, width_out, false,
                                         zero_ptr);
                    }
                } else {
                    for (int ci = 0; ci < oc_8; ++ci) {
                        if (bias_ptr) {
                            memcpy(bias_value, bias_ptr + ci * CH_PACK, CH_PACK * sizeof(float));
                        }
                        // trans output
                        float *dst_ci = dst_ptr + ci * oc_8_stride;
                        float *src_ci = src_ptr + ci * tile_count * CH_PACK;
                        output_trans_c8_post_2x4(src_ci, c_gi_stride, c_gi_stride * src_unit, src_trans_tmp_data,
                                                 CH_PACK, dst_unit * CH_PACK, bias_value, param->activation_type);
                        // copy to dest
                        memset(dst_trans_tmp_data, 0, 32 * sizeof(float));  // dst_unit * dst_unit * ch_pack
                        for (int i = 0; i < ey; ++i) {
                            memcpy(dst_trans_tmp_data + i * ex * CH_PACK, src_trans_tmp_data + i * CH_PACK * dst_unit,
                                   ex * sizeof(float) * CH_PACK);
                        }
                        unpack_output_c8(dst_trans_tmp_data, output_ptr, ci * CH_PACK, ci * CH_PACK + CH_PACK, dst_y,
                                         dst_y + ey, dst_x, dst_x + ex, channel_out, height_out, width_out, false,
                                         zero_ptr);
                    }
                }
            }
        }
    }

    _mm_free(pack_input);
    _mm_free(tmp_data);
    _mm_free(src_trans_tmp_data);
    _mm_free(dst_trans_tmp_data);

    return TNN_OK;
}

}  // namespace TNN_NS
