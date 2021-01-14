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

#include "tnn/device/x86/acc/convolution/x86_conv_layer_1x1.h"
#include "tnn/device/x86/x86_common.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/device/x86/x86_util.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
bool X86ConvLayer1x1::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                       const std::vector<Blob *> &outputs) {
    if (!param) {
        return false;
    }

    const int kw = param->kernels[0];
    const int kh = param->kernels[1];
    const int dw = param->dialations[0];
    const int dh = param->dialations[1];
    const int sw = param->strides[0];
    const int sh = param->strides[1];
    const int pw = param->pads[0] + param->pads[1];
    const int ph = param->pads[2] + param->pads[3];

    return kw == 1 && kh == 1 &&
           dw == 1 && dh == 1 &&
           sw == 1 && sh == 1 &&
           pw == 0 && ph == 0;
}

X86ConvLayer1x1::~X86ConvLayer1x1() {}

Status X86ConvLayer1x1::allocateBufferWeight(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    if (!buffer_weight_.GetBytesSize()) {

        const float *src = conv_res->filter_handle.force_to<float *>();
        int weight_count   = conv_res->filter_handle.GetDataCount();
        int data_byte_size = DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());

        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
            RawBuffer temp_buffer(weight_count * data_byte_size);
            float *dst = temp_buffer.force_to<float *>();

            memcpy(dst, src, data_byte_size * weight_count);
            temp_buffer.SetDataType(DATA_TYPE_FLOAT);
            buffer_weight_ = temp_buffer;
        } else {
            LOGE("Error: DataType %d not support\n", conv_res->filter_handle.GetDataType());
            return Status(TNNERR_MODEL_ERR, "conv_res DataType is not supported");
        }
    }
    return TNN_OK;
}

Status X86ConvLayer1x1::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);
    ConvLayerResource *resource = dynamic_cast<ConvLayerResource *>(resource_);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    const int batch    = dims_output[0];

    int dst_z_step     = dims_output[2] * dims_output[3];
    int src_z_step     = dims_input[2] * dims_input[3];

    float *weights_data = buffer_weight_.force_to<float*>();
    float *bias_data    = buffer_bias_.force_to<float*>();

    const float *src_origin = reinterpret_cast<const float *>(input->GetHandle().base);
    float *dst_origin = reinterpret_cast<float *>(output->GetHandle().base);

    // X86_matrixMul in row major format
    int m = dims_output[1];
    int n = src_z_step;
    int k = dims_input[1];

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        const float * B = src_origin + batch_idx * k * n;
        const float * A = weights_data;
        float * C = dst_origin + batch_idx * m * n;
        X86_matrixMul(m, n, k, A, B, C, 1, bias_data, param->activation_type);
    }

    return TNN_OK;
}

}  // namespace TNN_NS
