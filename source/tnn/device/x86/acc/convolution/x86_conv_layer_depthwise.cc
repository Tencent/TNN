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
#include "tnn/device/x86/x86_context.h"
#include "tnn/device/x86/x86_util.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"
#include <mm_malloc.h>

namespace TNN_NS {
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
        const int group4 = ROUND_UP(group, 4);

        const float *src = conv_res->filter_handle.force_to<float *>();

        int weight_count   = group4 * kh * kw;
        int data_byte_size = DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());

        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
            RawBuffer temp_buffer(weight_count * data_byte_size);
            float *dst = temp_buffer.force_to<float *>();

            PackC8(dst, src, kh * kw, kh * kw, kh * kw, group);
            temp_buffer.SetDataType(DATA_TYPE_FLOAT);
            buffer_weight_ = temp_buffer;
        } else {
            LOGE("Error: DataType %d not support\n", conv_res->filter_handle.GetDataType());
            return Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        }
    }
    return TNN_OK;
}

void PackC8WithPad(
    const float* src,
    float* dst,
    std::vector<int> pads,
    int src_h,
    int src_w,
    int channels) {

    int src_pad_w_stride = (src_w + pads[0] + pads[1]) * 8;
    memset(dst, 0, pads[2] * src_pad_w_stride * sizeof(float));

    auto dst_ptr = dst + pads[2] * src_pad_w_stride;
    for (int i = 0; i < src_h; i++) {
        auto dst_h_ptr = dst_ptr + i * src_pad_w_stride;
        auto src_h_ptr = src + i * src_w;
        memset(dst_h_ptr, 0, pads[0] * 8 * sizeof(float));
        PackC8(dst_h_ptr + pads[0] * 8, src_h_ptr, src_w, src_h * src_w, src_w, channels);
        memset(dst_h_ptr + pads[0] * 8 + src_w * 8, 0, pads[1] * 8 * sizeof(float));
    }
    memset(dst_ptr + src_h * src_pad_w_stride, 0, pads[3] * src_pad_w_stride * sizeof(float));
}

Status X86ConvLayerDepthwise::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);
    ConvLayerResource *resource = dynamic_cast<ConvLayerResource *>(resource_);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    const int batch    = dims_output[0];
    int dst_z_step     = dims_output[2] * dims_output[3];
    int src_z_step     = dims_input[2] * dims_input[3];
    int dilate_y_step  = (dims_input[3] + param->pads[0] + param->pads[1]) * 8 * param->dialations[1];
    int dilate_x_step  = 8 * param->dialations[0];
    int weight_z_step  = param->kernels[0] * param->kernels[1];
    int src_pad_w      = dims_input[3] + param->pads[0] + param->pads[1];

    size_t src_pad_size = (dims_input[3] + param->pads[0] + param->pads[1]) *
                          (dims_input[2] + param->pads[2] + param->pads[3]) *
                          8 * sizeof(float);
    size_t dst_tmp_size = dst_z_step * 8 * sizeof(float);
    // float *workspace = (float*)aligned_alloc(32, src_pad_size + dst_tmp_size);
    float *workspace = (float*)_mm_malloc(src_pad_size + dst_tmp_size, 32);

    const float *src_origin = reinterpret_cast<const float *>(input->GetHandle().base);
    float *dst_origin = reinterpret_cast<float *>(output->GetHandle().base);
    auto dw_full    = DepthwiseConvAVX2;

    float *weights_data = buffer_weight_.force_to<float*>();
    float *bias_data = nullptr;
    if (resource->bias_handle.GetDataCount() != 0) {
        bias_data = resource->bias_handle.force_to<float*>();
    }
    /*
    convdw3x3 stride >= 2 here
    convdw3x3s1 has separate kernel in convdws1 acc
    */
    // if (param->kernels[0] == 3 && param->kernels[1] == 3) {
    //     dw_full = DepthwiseConv3x3<T>;
    // }
    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto src_ptr = src_origin + batch_idx * dims_input[1] * src_z_step;
        auto dst_ptr = dst_origin + batch_idx * dims_output[1] * dst_z_step;

        // OMP_PARALLEL_FOR_
        int dz = 0;
        for (; dz + 7 < dims_output[1]; dz += 8) {
            auto *dst_z     = dst_ptr + dst_z_step * dz;
            auto *src_z     = src_ptr + src_z_step * dz;
            auto *weight_dz = weights_data + dz * weight_z_step;
            auto *bias_z    = bias_data + dz;
            auto *src_buf   = workspace;
            auto *dst_buf   = workspace + src_pad_size / sizeof(float);

            PackC8WithPad(src_z, src_buf, param->pads, dims_input[2], dims_input[3], 8);
            dw_full(dst_buf, src_buf, weight_dz, dims_output[3], param->strides[0] * 8,
                    param->kernels[0], param->kernels[1], dilate_x_step, dilate_y_step,
                    dims_output[2], src_pad_w * 8 * param->strides[1], dims_output[3] * 8);
            UnpackC8(dst_z, dst_buf, dst_z_step, dst_z_step, dst_z_step, 8);
        }
        if (dz < dims_output[1]) {
            auto *dst_z     = dst_ptr + dst_z_step * dz;
            auto *src_z     = src_ptr + src_z_step * dz;
            auto *weight_dz = weights_data + dz * weight_z_step;
            int left_c      = dims_output[1] - dz;

            float bias_tmp[8] = {0};
            memcpy(bias_tmp, bias_data + dz, left_c * sizeof(float));
            auto *src_buf   = workspace;
            auto *dst_buf   = workspace + src_pad_size / sizeof(float);

            PackC8WithPad(src_z, src_buf, param->pads, dims_input[2], dims_input[3], left_c);
            dw_full(dst_buf, src_buf, weight_dz, dims_output[3], param->strides[0] * 8,
                    param->kernels[0], param->kernels[1], dilate_x_step, dilate_y_step,
                    dims_output[2], src_pad_w * 8 * param->strides[1], dims_output[3] * 8);
            UnpackC8(dst_z, dst_buf, dst_z_step, dst_z_step, dst_z_step, left_c);
        }
    }

    // PostExec<T>(outputs);
    _mm_free(workspace);
    return TNN_OK;
}

}  // namespace TNN_NS
