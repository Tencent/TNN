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

#include "tnn/device/x86/acc/convolution/x86_conv_layer_depthwise.h"
#include "tnn/device/x86/x86_common.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/device/x86/x86_util.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
using namespace x86;

bool X86ConvLayerDepthwise::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                       const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    const int group          = param->group;
    const int input_channel  = inputs[0]->GetBlobDesc().dims[1];
    const int output_channel = outputs[0]->GetBlobDesc().dims[1];

    return group == input_channel && group == output_channel;
}

X86ConvLayerDepthwise::~X86ConvLayerDepthwise() {}

Status X86ConvLayerDepthwise::allocateBufferWeight(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    if (!buffer_weight_.GetBytesSize()) {
        int kw = param->kernels[0];
        int kh = param->kernels[1];

        const int group  = param->group;
        const float *src = conv_res->filter_handle.force_to<float *>();

        int group_rup = ROUND_UP(group, 8);
        if (arch_ == sse42) {
            group_rup = ROUND_UP(group, 4);
        }
        int weight_count   = group_rup * kh * kw;
        int data_byte_size = DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());

        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
            RawBuffer temp_buffer(weight_count * data_byte_size);
            float *dst = temp_buffer.force_to<float *>();

            if (arch_ == avx2) {
                PackC8(dst, src, kh * kw, kh * kw, kh * kw, group);
            } else if (arch_ == sse42) {
                PackC4(dst, src, kh * kw, kh * kw, kh * kw, group);
            }
            temp_buffer.SetDataType(DATA_TYPE_FLOAT);
            buffer_weight_ = temp_buffer;
        } else {
            LOGE("Error: DataType %d not support\n", conv_res->filter_handle.GetDataType());
            return Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        }
    }
    return TNN_OK;
}

template <int c_pack>
void PackWithPad(
    const float* src,
    float* dst,
    std::vector<int> pads,
    int src_h,
    int src_w,
    int channels) {

    auto PackAcc = PackC4;
    if (c_pack == 8) {
        PackAcc = PackC8;
    }
    int src_pad_w_stride = (src_w + pads[0] + pads[1]) * c_pack;
    memset(dst, 0, pads[2] * src_pad_w_stride * sizeof(float));

    auto dst_ptr = dst + pads[2] * src_pad_w_stride;
    for (int i = 0; i < src_h; i++) {
        auto dst_h_ptr = dst_ptr + i * src_pad_w_stride;
        auto src_h_ptr = src + i * src_w;
        memset(dst_h_ptr, 0, pads[0] * c_pack * sizeof(float));
        PackAcc(dst_h_ptr + pads[0] * c_pack, src_h_ptr, src_w, src_h * src_w, src_w, channels);
        memset(dst_h_ptr + pads[0] * c_pack + src_w * c_pack, 0, pads[1] * c_pack * sizeof(float));
    }
    memset(dst_ptr + src_h * src_pad_w_stride, 0, pads[3] * src_pad_w_stride * sizeof(float));
}

Status X86ConvLayerDepthwise::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    int c_pack = 8;
    if (arch_ == sse42) {
        c_pack = 4;
    }

    const int batch    = dims_output[0];
    int dst_z_step     = dims_output[2] * dims_output[3];
    int src_z_step     = dims_input[2] * dims_input[3];
    int src_pad_w      = dims_input[3] + param->pads[0] + param->pads[1];
    int dilate_y_step  = src_pad_w * c_pack * param->dialations[1];
    int dilate_x_step  = c_pack * param->dialations[0];
    int weight_z_step  = param->kernels[0] * param->kernels[1];

    int max_num_threads = OMP_MAX_THREADS_NUM_;
    size_t src_pad_size = ROUND_UP(src_pad_w * (dims_input[2] + param->pads[2] + param->pads[3]) * c_pack * sizeof(float), 32);
    size_t dst_tmp_size = ROUND_UP(dst_z_step * c_pack * sizeof(float), 32);
    float *workspace = reinterpret_cast<float *>(context_->GetSharedWorkSpace(
                                                 (src_pad_size + dst_tmp_size) * max_num_threads));

    const float *src_origin = handle_ptr<const float *>(input->GetHandle());
    float *dst_origin = handle_ptr<float *>(output->GetHandle());

    auto dw_full = DepthwiseConv<ActivationType_None, Float8, 8>;
    if (param->activation_type == ActivationType_ReLU) {
        dw_full  = DepthwiseConv<ActivationType_ReLU, Float8, 8>;
    } else if (param->activation_type == ActivationType_ReLU6) {
        dw_full  = DepthwiseConv<ActivationType_ReLU6, Float8, 8>;
    }
    if (arch_ == sse42) {
        dw_full = DepthwiseConv<ActivationType_None, Float4, 4>;
        if (param->activation_type == ActivationType_ReLU) {
            dw_full  = DepthwiseConv<ActivationType_ReLU, Float4, 4>;
        } else if (param->activation_type == ActivationType_ReLU6) {
            dw_full  = DepthwiseConv<ActivationType_ReLU6, Float4, 4>;
        }
    }

    auto PackWithPadAcc = PackWithPad<8>;
    if (arch_ == sse42) {
        PackWithPadAcc = PackWithPad<4>;
    }

    auto UnpackAcc = UnpackC8;
    if (arch_ == sse42) {
        UnpackAcc = UnpackC4;
    }

    float *weights_data = buffer_weight_.force_to<float*>();
    float *bias_data = buffer_bias_.force_to<float*>();;

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto src_ptr = src_origin + batch_idx * dims_input[1] * src_z_step;
        auto dst_ptr = dst_origin + batch_idx * dims_output[1] * dst_z_step;

        OMP_PARALLEL_FOR_GUIDED_
        for (int dz = 0; dz < dims_output[1]; dz += c_pack) {
            int real_dz     = MIN(c_pack, dims_output[1] - dz);
            auto *dst_z     = dst_ptr + dst_z_step * dz;
            auto *src_z     = src_ptr + src_z_step * dz;
            auto *weight_dz = weights_data + dz * weight_z_step;
            auto *bias_z    = bias_data + dz;
            int thread_id   = OMP_TID_;
            auto *tmp_buf   = workspace + thread_id * ((src_pad_size + dst_tmp_size) / sizeof(float));
            auto *src_buf   = tmp_buf;
            auto *dst_buf   = tmp_buf + src_pad_size / sizeof(float);

            PackWithPadAcc(src_z, src_buf, param->pads, dims_input[2], dims_input[3], real_dz);
            dw_full(dst_buf, src_buf, weight_dz, bias_z, dims_output[3], param->strides[0] * c_pack,
                    param->kernels[0], param->kernels[1], dilate_x_step, dilate_y_step,
                    dims_output[2], src_pad_w * c_pack * param->strides[1], dims_output[3] * c_pack);
            UnpackAcc(dst_z, dst_buf, dst_z_step, dst_z_step, dst_z_step, real_dz);
        }
    }
    return TNN_OK;
}

}  // namespace TNN_NS
