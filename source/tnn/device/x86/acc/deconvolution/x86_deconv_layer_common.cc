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

#include "tnn/device/x86/acc/deconvolution/x86_deconv_layer_common.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {
/*
X86DeconvLayerCommonas as the last solution, always return true
handle the case group != 1, dilate != 1, any pads and strides
*/
bool X86DeconvLayerCommon::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    return true;
}

X86DeconvLayerCommon::~X86DeconvLayerCommon() {}

Status X86DeconvLayerCommon::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status X86DeconvLayerCommon::allocateBufferWeight(const std::vector<Blob *> &inputs,
                                                  const std::vector<Blob *> &outputs) {
    ConvLayerParam *param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    if (!buffer_weight_.GetBytesSize()) {
        int k_c     = conv_gemm_conf_.K_c_;
        int m_c     = conv_gemm_conf_.M_c_;
        int n_block = conv_gemm_conf_.n_block_;

        int K = dims_input[1] / param->group;
        int M = dims_output[1] * param->kernels[0] * param->kernels[1] / param->group;

        size_t weight_pack_per_group = ROUND_UP(K, k_c) * ROUND_UP(M, n_block);

        RawBuffer transpose_buffer(conv_res->filter_handle.GetBytesSize() / param->group);
        const float *src = conv_res->filter_handle.force_to<float *>();

        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
            float *trans = transpose_buffer.force_to<float *>();
            RawBuffer temp_buffer(weight_pack_per_group * param->group * sizeof(float));
            float *dst = temp_buffer.force_to<float *>();

            for (int g = 0; g < param->group; g++) {
                auto src_g = src + K * M * g;
                MatTranspose(trans, src_g, K, M);
                auto dst_g = dst + weight_pack_per_group * g;
                conv_pack_weights(M, K, trans, K, dst_g, conv_gemm_conf_);
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

Status X86DeconvLayerCommon::allocateBufferBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    if (!buffer_bias_.GetBytesSize()) {
        auto dims_output = outputs[0]->GetBlobDesc().dims;
        int total_byte_size =
            ROUND_UP(dims_output[1], 8) * DataTypeUtils::GetBytesSize(conv_res->bias_handle.GetDataType());
        RawBuffer temp_buffer(total_byte_size);
        if (conv_param->bias) {
            const int bias_handle_size    = conv_res->bias_handle.GetBytesSize();
            const float *bias_handle_data = conv_res->bias_handle.force_to<float *>();
            memcpy(temp_buffer.force_to<float *>(), conv_res->bias_handle.force_to<float *>(), bias_handle_size);
        }
        buffer_bias_ = temp_buffer;
    }
    return TNN_OK;
}

Status X86DeconvLayerCommon::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param);
    auto status                = X86LayerAcc::Init(context, param, resource, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    conv_gemm_conf_ = conv_gemm_config<float, float, float>();

    RETURN_ON_NEQ(allocateBufferWeight(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferBias(inputs, outputs), TNN_OK);

    switch (conv_param->activation_type) {
        case ActivationType_None:
            post_func_ = (arch_ == avx2) ? X86_Post_Exec<ActivationType_None, Float8, 8>
                                         : X86_Post_Exec<ActivationType_None, Float4, 4>;
            break;
        case ActivationType_ReLU:
            post_func_ = (arch_ == avx2) ? X86_Post_Exec<ActivationType_ReLU, Float8, 8>
                                         : X86_Post_Exec<ActivationType_ReLU, Float4, 4>;
            break;
        case ActivationType_ReLU6:
            post_func_ = (arch_ == avx2) ? X86_Post_Exec<ActivationType_ReLU6, Float8, 8>
                                         : X86_Post_Exec<ActivationType_ReLU6, Float4, 4>;
            break;
        default:
            break;
    }

    return TNN_OK;
}

Status X86DeconvLayerCommon::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto input_dims   = inputs[0]->GetBlobDesc().dims;
    auto output_dims  = outputs[0]->GetBlobDesc().dims;
    void *input_ptr   = input_blob->GetHandle().base;
    void *output_ptr  = output_blob->GetHandle().base;
    auto param        = dynamic_cast<ConvLayerParam *>(param_);
    auto resource     = dynamic_cast<ConvLayerResource *>(resource_);

    int conv_out_offset_      = output_dims[2] * output_dims[3] * output_dims[1];
    int conv_in_spatial_dim_  = input_dims[2] * input_dims[3];
    int conv_out_spatial_dim_ = output_dims[2] * output_dims[3];
    int input_offset_         = input_dims[1] * conv_in_spatial_dim_ / param->group;
    size_t col_offset_ =
        param->kernels[0] * param->kernels[1] * input_dims[2] * input_dims[3] * (output_dims[1] / param->group);

    int m_c               = conv_gemm_conf_.M_c_;
    int k_c               = conv_gemm_conf_.K_c_;
    int n_block           = conv_gemm_conf_.n_block_;
    size_t src_trans_size = m_c * k_c;

    size_t im2col_size    = ROUND_UP(col_offset_ * param->group * sizeof(float), 32);
    size_t workspace_size = (im2col_size + ROUND_UP(src_trans_size * sizeof(float), 32));
    float *workspace      = reinterpret_cast<float *>(context_->GetSharedWorkSpace(workspace_size));

    float *im2col_workspace    = workspace;
    float *src_trans_workspace = workspace + im2col_size / sizeof(float);

    int K = input_dims[1] / param->group;
    int M = output_dims[1] * param->kernels[0] * param->kernels[1] / param->group;
    int N = conv_in_spatial_dim_;

    size_t weight_offset_per_group = ROUND_UP(K, k_c) * ROUND_UP(M, n_block);

    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto input_data   = static_cast<float *>(input_ptr);
        auto output_data  = static_cast<float *>(output_ptr);
        auto weights_data = buffer_weight_.force_to<float *>();
        float *bias_data  = buffer_bias_.force_to<float *>();
        RawBuffer fake_bias(ROUND_UP(M, 8) * sizeof(float));
        for (size_t b = 0; b < output_dims[0]; b++) {
            for (int g = 0; g < param->group; g++) {
                conv_sgemm_nn_col_major(N, M, K, input_data + (b * param->group + g) * input_offset_, N,
                                        weights_data + weight_offset_per_group * g, K,
                                        im2col_workspace + col_offset_ * g, N, fake_bias.force_to<float *>(), 0,
                                        src_trans_workspace, conv_gemm_conf_);
            }

            X86_COL2IM(im2col_workspace, output_dims[1], input_dims[2], input_dims[3], param->kernels[1],
                       param->kernels[0], param->pads[2], param->pads[0], param->strides[1], param->strides[0],
                       param->dialations[1], param->dialations[0], output_dims[2], output_dims[3],
                       output_data + b * conv_out_offset_);

            if (post_func_) {
                post_func_(output_data + b * conv_out_offset_, bias_data, output_dims[1], conv_out_spatial_dim_);
            }
        }
    } else {
        return Status(TNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT, "Error: x86 device not support this data type");
    }

    return TNN_OK;
}
}  // namespace TNN_NS
