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

#include "tnn/device/cpu/acc/cpu_conv_layer_acc.h"

#include "tnn/core/blob_int8.h"
#include "tnn/interpreter/layer_resource_generator.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

CpuConvLayerAcc::~CpuConvLayerAcc() {}

Status CpuConvLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                             const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    CPU_CONVERT_HALF_RESOURCE(LAYER_CONVOLUTION);
    if (runtime_model_ != RUNTIME_MODE_NORMAL) {
        return TNN_OK;
    }

    auto conv_param = dynamic_cast<ConvLayerParam *>(param);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource* conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);
    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        if (!buffer_scale_.GetBytesSize()) {
            auto dims_output    = outputs[0]->GetBlobDesc().dims;
            int total_byte_size = dims_output[1] * sizeof(float);

            const float *w_scale = conv_res->scale_handle.force_to<float *>();
            CHECK_PARAM_NULL(w_scale);

            const float *o_scale =
                reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.force_to<float *>();
            int scale_len_w = conv_res->scale_handle.GetDataCount();
            int scale_len_o = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.GetDataCount();
            RawBuffer temp_buffer(total_byte_size);
            float *temp_ptr = temp_buffer.force_to<float *>();
            int32_t *bias_ptr = conv_res->bias_handle.force_to<int32_t *>();
            for (int i = 0; i < dims_output[1]; i++) {
                int w_scale_idx = scale_len_w == 1 ? 0 : i;
                int o_scale_idx = scale_len_o == 1 ? 0 : i;
                if (o_scale[o_scale_idx] >= FLT_MIN)
                    temp_ptr[i] = w_scale[w_scale_idx] / o_scale[o_scale_idx];
                else
                    temp_ptr[i] = 0.0;
            }
            buffer_scale_ = temp_buffer;
        }
        if (!buffer_weight_x_bias_.GetBytesSize()) {
            do_bias_preprocess_  = true;
            auto dims_output     = outputs[0]->GetBlobDesc().dims;
            auto dims_input      = inputs[0]->GetBlobDesc().dims;
            int zero_point_len_w = conv_res->zero_point_handle.GetDataCount();
            int data_len_w       = conv_res->filter_handle.GetDataCount();
            int zero_point_len_i =
                reinterpret_cast<BlobInt8 *>(inputs[0])->GetIntResource()->zero_point_handle.GetDataCount();
            int8_t *zero_point_i_ptr =
                reinterpret_cast<BlobInt8 *>(inputs[0])->GetIntResource()->zero_point_handle.force_to<int8_t *>();
            int8_t *zero_point_w_ptr = conv_res->zero_point_handle.force_to<int8_t *>();
            int8_t *weight_ptr       = conv_res->filter_handle.force_to<int8_t *>();
            RawBuffer temp_buffer_weight_x_bias(dims_output[1] * sizeof(int32_t));
            int32_t *weight_x_bias_ptr    = temp_buffer_weight_x_bias.force_to<int32_t *>();
            int kernel_len                = conv_param->kernels[1] * conv_param->kernels[0];
            int group                     = conv_param->group;
            int input_channels_per_group  = dims_input[1] / group;
            int output_channels_per_group = dims_output[1] / group;

            for (int g = 0; g < group; ++g) {
                int weights_start  = g * data_len_w / group;
                int output_c_start = g * output_channels_per_group;
                int output_c_end   = (g + 1) * output_channels_per_group;
                int input_c_start  = g * input_channels_per_group;
                int input_c_end    = (g + 1) * input_channels_per_group;
                for (int output_c = output_c_start; output_c < output_c_end; output_c++) {
                    int w_zero_point_idx = zero_point_len_w == 1 ? 0 : output_c;
                    int32_t val          = 0;
                    for (int input_c = input_c_start; input_c < input_c_end; input_c++) {
                        int i_zero_point_idx = zero_point_len_i == 1 ? 0 : input_c;
                        int kernel_sum       = 0;
                        int8_t *kernel_ptr   = weight_ptr + weights_start +
                                             kernel_len * (((output_c - output_c_start) * input_channels_per_group) +
                                                           input_c - input_c_start);
                        std::for_each(kernel_ptr, kernel_ptr + kernel_len, [&](int8_t n) { kernel_sum += n; });
                        val += zero_point_i_ptr[i_zero_point_idx] *
                               (zero_point_w_ptr[w_zero_point_idx] * kernel_len - kernel_sum);
                    }
                    weight_x_bias_ptr[output_c] = val;
                }
            }
            buffer_weight_x_bias_ = temp_buffer_weight_x_bias;
        }
        if (conv_param->fusion_type != FusionType_None && !buffer_add_scale_.GetBytesSize()) {
            auto dims_output    = outputs[0]->GetBlobDesc().dims;
            int total_byte_size = dims_output[1] * sizeof(float);

            auto add_input_resource  = reinterpret_cast<BlobInt8 *>(inputs[1])->GetIntResource();
            auto add_output_resource = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource();

            const float *i_scale = add_input_resource->scale_handle.force_to<float *>();

            const float *o_scale = add_output_resource->scale_handle.force_to<float *>();

            int scale_len_i = add_input_resource->scale_handle.GetDataCount();
            int scale_len_o = add_output_resource->scale_handle.GetDataCount();
            RawBuffer temp_buffer(total_byte_size);
            float *temp_ptr = temp_buffer.force_to<float *>();
            for (int i = 0; i < dims_output[1]; i++) {
                int scale_idx_i = scale_len_i == 1 ? 0 : i;
                int scale_idx_o = scale_len_o == 1 ? 0 : i;

                if (o_scale[scale_idx_o] >= FLT_MIN)
                    temp_ptr[i] = i_scale[scale_idx_i] / o_scale[scale_idx_o];
                else
                    temp_ptr[i] = 0.0;
            }
            buffer_add_scale_ = temp_buffer;
        }
        if (conv_param->activation_type == ActivationType_ReLU6) {
            auto output_scale_resource      = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource();
            auto output_scale_len           = output_scale_resource->scale_handle.GetDataCount();
            auto output_scale_resource_data = output_scale_resource->scale_handle.force_to<float *>();
            auto &dims_output               = outputs[0]->GetBlobDesc().dims;
            auto &output_channel            = dims_output[1];
            RawBuffer relu6_max             = RawBuffer(output_channel * sizeof(int8_t));
            auto relu6_max_data             = relu6_max.force_to<int8_t *>();
            for (int i = 0; i < output_channel; ++i) {
                int scale_idx     = output_scale_len == 1 ? 0 : i;
                relu6_max_data[i] = float2int8(6.0f / output_scale_resource_data[scale_idx]);
            }
            relu6_max_ = relu6_max;
            relu6_max_.SetDataType(DATA_TYPE_INT8);
        }
    }
    return TNN_OK;
}

Status CpuConvLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuConvLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param    = dynamic_cast<ConvLayerParam *>(param_);
    ConvLayerResource* resource = dynamic_cast<ConvLayerResource *>(resource_);

    if (!param || !resource) {
        return Status(TNNERR_MODEL_ERR, "Error: ConvLayerParam or ConvLayerResource is empty");
    }

    Blob *input_blob   = inputs[0];
    Blob *output_blob  = outputs[0];
    void *input_ptr    = input_blob->GetHandle().base;
    void *output_ptr   = output_blob->GetHandle().base;
    void *weight_ptr   = resource->filter_handle.force_to<void *>();
    void *bias_ptr     = NULL;
    DataType data_type = output_blob->GetBlobDesc().data_type;
    if (param->bias || data_type == DATA_TYPE_INT8) {
        bias_ptr = resource->bias_handle.force_to<void *>();
    }
    DimsVector output_dims = output_blob->GetBlobDesc().dims;
    DimsVector input_dims  = input_blob->GetBlobDesc().dims;

    if (data_type == DATA_TYPE_FLOAT) {
        NaiveConv<float, float, float, float>(input_ptr, output_ptr, weight_ptr, bias_ptr, input_dims, output_dims,
                                              param->strides[1], param->strides[0], param->kernels[1],
                                              param->kernels[0], param->pads[2], param->pads[0], param->group,
                                              param->dialations[1], param->activation_type, NULL, 0, NULL, 0);
    } else if (data_type == DATA_TYPE_BFP16) {
        NaiveConv<bfp16_t, float, float, bfp16_t>(input_ptr, output_ptr, weight_ptr, bias_ptr, input_dims, output_dims,
                                                  param->strides[1], param->strides[0], param->kernels[1],
                                                  param->kernels[0], param->pads[2], param->pads[0], param->group,
                                                  param->dialations[1], param->activation_type, NULL, 0, NULL, 0);
    } else if (data_type == DATA_TYPE_INT8) {
        auto weight_scale = buffer_scale_.force_to<float *>();
        int zero_point_len_w = resource->zero_point_handle.GetDataCount();
        int zero_point_len_i = reinterpret_cast<BlobInt8 *>(input_blob)->GetIntResource()->zero_point_handle.GetDataCount();
        int zero_point_len_o = reinterpret_cast<BlobInt8 *>(output_blob)->GetIntResource()->zero_point_handle.GetDataCount();
        int8_t* zero_point_w_ptr = resource->zero_point_handle.force_to<int8_t *>();
        int8_t* zero_point_i_ptr = reinterpret_cast<BlobInt8 *>(input_blob)->GetIntResource()->zero_point_handle.force_to<int8_t *>();
        int8_t* zero_point_o_ptr = reinterpret_cast<BlobInt8 *>(output_blob)->GetIntResource()->zero_point_handle.force_to<int8_t *>();
        auto relu6_max    = relu6_max_.force_to<int8_t *>();
        void *add_input   = (param->fusion_type == FusionType_None) ? nullptr : inputs[1]->GetHandle().base;
        int8_t *add_bias_input = (param->fusion_type == FusionType_None) ? nullptr
                : reinterpret_cast<BlobInt8 *>(inputs[1])->GetIntResource()->zero_point_handle.force_to<int8_t *>();
        if (do_bias_preprocess_) {
            NaiveConvBias<int8_t, int8_t, int32_t, int8_t>(
                input_ptr, output_ptr, weight_ptr, bias_ptr, input_dims, output_dims, param->strides[1],
                param->strides[0], param->kernels[1], param->kernels[0], param->pads[2], param->pads[0], param->group,
                param->dialations[1], param->activation_type, weight_scale, buffer_scale_.GetDataCount(),
                zero_point_w_ptr, zero_point_len_w, zero_point_i_ptr, zero_point_len_i, zero_point_o_ptr,
                zero_point_len_o, buffer_weight_x_bias_.force_to<int32_t *>(), relu6_max, relu6_max_.GetDataCount(),
                param->fusion_type, add_input, buffer_add_scale_.force_to<float *>(), add_bias_input);
        } else {
            NaiveConvBias<int8_t, int8_t, int32_t, int8_t>(
                input_ptr, output_ptr, weight_ptr, bias_ptr, input_dims, output_dims, param->strides[1],
                param->strides[0], param->kernels[1], param->kernels[0], param->pads[2], param->pads[0], param->group,
                param->dialations[1], param->activation_type, weight_scale, buffer_scale_.GetDataCount(),
                zero_point_w_ptr, zero_point_len_w, zero_point_i_ptr, zero_point_len_i, zero_point_o_ptr,
                zero_point_len_o, relu6_max, relu6_max_.GetDataCount(), param->fusion_type, add_input,
                buffer_add_scale_.force_to<float *>(), add_bias_input);
        }
    } else {
        return Status(TNNERR_LAYER_ERR, "data type not support in conv");
    }
    return TNN_OK;
}

CpuTypeLayerAccRegister<TypeLayerAccCreator<CpuConvLayerAcc>> g_cpu_conv_layer_acc_register(LAYER_CONVOLUTION);

}  // namespace TNN_NS
