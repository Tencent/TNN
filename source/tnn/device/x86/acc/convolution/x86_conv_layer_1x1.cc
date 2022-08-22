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

    if (param->group != 1) {
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

Status X86ConvLayer1x1::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    const int batch    = dims_output[0];

    int dst_z_step     = dims_output[2] * dims_output[3];
    int src_z_step     = dims_input[2] * dims_input[3];

    float *weights_data = buffer_weight_.force_to<float*>();
    float *bias_data    = buffer_bias_.force_to<float*>();

    const float *src_origin = handle_ptr<const float *>(input->GetHandle());
    float *dst_origin = handle_ptr<float *>(output->GetHandle());

    // X86_matrixMul in row major format
    int m = dims_output[1];
    int n = src_z_step;
    int k = dims_input[1];

    int max_num_threads = OMP_MAX_THREADS_NUM_;
    conv_ajust_m_blk_size(max_num_threads, src_z_step, conv_gemm_conf_.M_c_);

    int m_c = conv_gemm_conf_.M_c_;
    int k_c = conv_gemm_conf_.K_c_;

    float *src_buf = reinterpret_cast<float *>(
        context_->GetSharedWorkSpace(m_c * k_c * max_num_threads * sizeof(float)));

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        const float * B = src_origin + batch_idx * k * n;
        const float * A = weights_data;
        float * C = dst_origin + batch_idx * m * n;

        conv_sgemm_nn_col_major_prepack_b(n, m, k, B, n, A, k, C, n,
            bias_data, param->activation_type, src_buf, conv_gemm_conf_);
    }

    return TNN_OK;
}

}  // namespace TNN_NS
