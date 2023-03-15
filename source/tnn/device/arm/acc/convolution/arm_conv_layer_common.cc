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

#include "tnn/device/arm/acc/convolution/arm_conv_layer_common.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

#if defined(__aarch64__)
#define CONVOLUTION_TILED_NUMBER (14)
#else
#define CONVOLUTION_TILED_NUMBER (8)
#endif

namespace TNN_NS {
/*
ArmConvLayerCommonas as the last solution, always return true
handle the case group != 1, dilate != 1, any pads and strides
*/
bool ArmConvLayerCommon::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                    const std::vector<Blob *> &outputs) {
    return true;
}

ArmConvLayerCommon::~ArmConvLayerCommon() {}

Status ArmConvLayerCommon::allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    if (!buffer_weight_.GetBytesSize()) {
        int kw = conv_param->kernels[0];
        int kh = conv_param->kernels[1];

        const int group          = conv_param->group;
        const int input_channel  = dims_input[1];
        const int output_channel = dims_output[1];
        const int goc            = output_channel / group;
        const int gic            = input_channel / group;
        const int goc_4          = UP_DIV(goc, 4);
        const int gic_4          = UP_DIV(gic, 4);

        const float *src = conv_res->filter_handle.force_to<float *>();

        int weight_count   = group * goc_4 * gic_4 * kh * kw * 16;
        int data_byte_size = DataTypeUtils::GetBytesSize(conv_res->filter_handle.GetDataType());

        /*
        [ATTENTION]
        alloc more NEON_KERNEL_EXTRA_LOAD bytes for assemble kernel prefetch
        */
        RawBuffer temp_buffer(weight_count * data_byte_size + NEON_KERNEL_EXTRA_LOAD);
        float *dst = temp_buffer.force_to<float *>();

        ConvertWeightsFromGOIHWToGOIHW16((float *)src, (float *)dst, group, input_channel, output_channel,
                                         conv_param->kernels[1], conv_param->kernels[0]);

        buffer_weight_ = temp_buffer;
    }
    return TNN_OK;
}

Status ArmConvLayerCommon::allocateBufferBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    if (!buffer_bias_.GetBytesSize()) {
        auto dims_output = outputs[0]->GetBlobDesc().dims;
        int total_byte_size =
            ROUND_UP(dims_output[1], 4) * DataTypeUtils::GetBytesSize(conv_res->bias_handle.GetDataType());
        RawBuffer temp_buffer(total_byte_size);
        if (conv_param->bias) {
            const int bias_handle_size    = conv_res->bias_handle.GetBytesSize();
            const float *bias_handle_data = conv_res->bias_handle.force_to<float *>();

            if (conv_res->bias_handle.GetDataType() == DATA_TYPE_FLOAT ||
                conv_res->bias_handle.GetDataType() == DATA_TYPE_HALF) {
                memcpy(temp_buffer.force_to<float *>(), conv_res->bias_handle.force_to<float *>(), bias_handle_size);
            }
        }
        buffer_bias_ = temp_buffer;
    }

    return TNN_OK;
}

Status ArmConvLayerCommon::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);

    RETURN_ON_NEQ(allocateBufferWeight(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferBias(inputs, outputs), TNN_OK);

    k_param_->fil_ptr = buffer_weight_.force_to<void *>();
    k_param_->bias    = buffer_bias_.force_to<void *>();

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    /*
    post func, support bias, bias + relu/relu6
    */
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        if (conv_param->activation_type == ActivationType_ReLU) {
            post_func_ = PostAddBiasRelu<float>;
        } else if (conv_param->activation_type == ActivationType_SIGMOID_MUL) {
            post_func_ = context_->GetPrecision() == PRECISION_HIGH ? PostAddBiasSwish<float, float, false>
                                                                    : PostAddBiasSwish<float, float, true>;
        } else if (conv_param->activation_type == ActivationType_ReLU6) {
            post_func_ = PostAddBiasRelu6<float>;
        } else {
            post_func_ = PostAddBias<float>;
        }
    } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        if (conv_param->activation_type == ActivationType_ReLU) {
            post_func_ = PostAddBiasRelu<bfp16_t>;
        } else if (conv_param->activation_type == ActivationType_SIGMOID_MUL) {
            post_func_ = context_->GetPrecision() == PRECISION_HIGH ? PostAddBiasSwish<bfp16_t, float, false>
                                                                    : PostAddBiasSwish<bfp16_t, float, true>;
        } else if (conv_param->activation_type == ActivationType_ReLU6) {
            post_func_ = PostAddBiasRelu6<bfp16_t>;
        } else {
            post_func_ = PostAddBias<bfp16_t>;
        }
    }

    return TNN_OK;
}

Status ArmConvLayerCommon::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    }
    return TNNERR_LAYER_ERR;
}

template <typename T>
Status ArmConvLayerCommon::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);

    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    const int batch = dims_output[0];
    const int group = conv_param->group;
    auto ic = dims_input[1], input_slice = UP_DIV(dims_input[1], 4);
    auto oc = dims_output[1], output_slice = UP_DIV(dims_output[1], 4), output_slice_per_group = output_slice / group;

    auto gic                    = dims_input[1] / group;
    auto goc                    = dims_output[1] / group;
    auto gic_4                  = UP_DIV(gic, 4);
    auto goc_4                  = UP_DIV(goc, 4);
    auto input_bytes_per_group  = k_param_->iw * k_param_->ih * gic_4 * 4 * data_byte_size;
    auto output_bytes_per_group = k_param_->ow * k_param_->oh * goc_4 * 4 * data_byte_size;

    int dilate_y_step = k_param_->iw * 4 * conv_param->dialations[1];
    int dilate_x_step = 4 * conv_param->dialations[0];

    int src_z_step    = k_param_->iw * k_param_->ih * 4;
    int weight_z_step = conv_param->kernels[1] * conv_param->kernels[0] * gic_4 * 16;

    T *input_orign = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    T *dst_origin  = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    int max_num_threads = OMP_MAX_THREADS_NUM_;

    int x_count = UP_DIV(k_param_->ow, CONVOLUTION_TILED_NUMBER);
    int src_xc  = 1 + (CONVOLUTION_TILED_NUMBER - 1) * conv_param->strides[0] +
                 conv_param->dialations[0] * (conv_param->kernels[0] - 1);
    int workspace_per_thread = src_xc * conv_param->kernels[1] * ROUND_UP(dims_input[1], 4) * data_byte_size;
    RawBuffer i_buffer;
    RawBuffer o_buffer;

    T *work_space = reinterpret_cast<T *>(context_->GetSharedWorkSpace(max_num_threads * workspace_per_thread));

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        T *input_ptr;
        T *output_ptr;

        /*
        first unpack input tensor to nchw data format
        pack data to make sure every group channel algin4
        */
        if (gic_4 != (gic / 4) && group != 1) {
            RawBuffer i_temp_buffer_(group * input_bytes_per_group);
            RawBuffer temp_buffer(group * input_bytes_per_group);
            i_buffer  = i_temp_buffer_;
            input_ptr = i_buffer.force_to<T *>();

            UnpackC4(temp_buffer.force_to<T *>(),
                     input_orign + batch_idx * k_param_->iw * k_param_->ih * ROUND_UP(ic, 4),
                     k_param_->iw * k_param_->ih, ic);
            for (int g = 0; g < group; g++) {
                PackC4(input_ptr + g * input_bytes_per_group / 4,
                       temp_buffer.force_to<T *>() + g * k_param_->iw * k_param_->ih * gic, k_param_->iw * k_param_->ih,
                       gic);
            }
        } else {
            input_ptr = input_orign + batch_idx * k_param_->iw * k_param_->ih * ROUND_UP(ic, 4);
        }

        if (goc_4 != (goc / 4) && group != 1) {
            RawBuffer o_temp_buffer_(group * output_bytes_per_group);
            o_buffer   = o_temp_buffer_;
            output_ptr = o_buffer.force_to<T *>();
        } else {
            output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * ROUND_UP(oc, 4);
        }

        for (int g = 0; g < group; g++) {
            auto input_g_ptr  = input_ptr + g * k_param_->iw * k_param_->ih * gic_4 * 4;
            auto output_g_ptr = output_ptr + g * k_param_->ow * k_param_->oh * goc_4 * 4;
            auto w_g_offset   = g * goc_4 * weight_z_step;
            OMP_PARALLEL_FOR_
            for (int x = 0; x < x_count; x++) {
                int thread_id = OMP_TID_;

                auto work_space_t = work_space + thread_id * workspace_per_thread / sizeof(T);

                int x_idx    = (int)x * CONVOLUTION_TILED_NUMBER;
                int x_remain = k_param_->ow - x_idx;
                int x_c      = x_remain > CONVOLUTION_TILED_NUMBER ? CONVOLUTION_TILED_NUMBER : x_remain;
                int src_xc =
                    1 + (x_c - 1) * conv_param->strides[0] + conv_param->dialations[0] * (conv_param->kernels[0] - 1);
                int d_x         = x_idx;
                int src_start_x = d_x * conv_param->strides[0] - conv_param->pads[0];
                int src_end_x   = src_start_x + src_xc >= k_param_->iw ? k_param_->iw : src_start_x + src_xc;

                int dst_offset = 0;
                if (src_start_x < 0) {
                    dst_offset  = -src_start_x;
                    src_start_x = 0;
                }
                int copy_count = src_end_x - src_start_x;
                auto src_x     = input_g_ptr + 4 * src_start_x;

                for (int dy = 0; dy < k_param_->oh; dy++) {
                    /*
                    copy make board, data in workspace are dirty, should be clear first
                    */
                    memset(work_space_t, 0, workspace_per_thread);
                    int src_start_y = dy * conv_param->strides[1] - conv_param->pads[2];
                    int sfy         = MAX(0, (UP_DIV(-src_start_y, conv_param->dialations[1])));
                    int efy =
                        MIN(conv_param->kernels[1], UP_DIV(k_param_->ih - src_start_y, conv_param->dialations[1]));

                    for (int sz = 0; sz < gic_4; sz++) {
                        auto dst_z = work_space_t + sz * src_xc * conv_param->kernels[1] * 4;
                        auto src_z = src_x + sz * src_z_step;
                        for (int ky = sfy; ky < efy; ky++) {
                            int sy     = src_start_y + ky * conv_param->dialations[1];
                            auto src_y = src_z + 4 * sy * k_param_->iw;
                            auto dst_y = dst_z + (ky * src_xc + dst_offset) * 4;
                            memcpy(dst_y, src_y, copy_count * 4 * sizeof(T));
                        }
                    }

                    // output: tile x oc
                    for (int dz = 0; dz < goc_4; dz++) {
                        auto dst_z =
                            output_g_ptr + dz * k_param_->ow * k_param_->oh * 4 + x_idx * 4 + k_param_->ow * 4 * dy;
                        auto weight_dz = reinterpret_cast<float *>(k_param_->fil_ptr) + w_g_offset + dz * weight_z_step;

                        ConvCommonO4(dst_z, work_space_t, weight_dz, x_c, conv_param->strides[0] * 4, gic_4,
                                     src_xc * 4 * conv_param->kernels[1], conv_param->kernels[0],
                                     conv_param->kernels[1], dilate_x_step, src_xc * 4);
                    }
                }
            }
        }

        /*
        first unpack every group output data to get nchw data format
        pack data to make sure output tensor channel algin4 and continuously
        */
        if (goc_4 != (goc / 4) && group != 1) {
            RawBuffer temp_buffer(group * output_bytes_per_group);
            for (int g = 0; g < group; g++) {
                UnpackC4(temp_buffer.force_to<T *>() + g * k_param_->ow * k_param_->oh * goc,
                         output_ptr + g * k_param_->ow * k_param_->oh * goc_4 * 4, k_param_->ow * k_param_->oh, goc);
            }
            PackC4(dst_origin + batch_idx * k_param_->ow * k_param_->oh * ROUND_UP(oc, 4), temp_buffer.force_to<T *>(),
                   k_param_->ow * k_param_->oh, oc);
        }
    }

    PostExec<T>(outputs);

    return TNN_OK;
}

template <typename T>
void ArmConvLayerCommon::PostExec(const std::vector<Blob *> &outputs) {
    const int batch = outputs[0]->GetBlobDesc().dims[0];
    auto dst_origin = reinterpret_cast<T *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
    if (post_func_) {
        OMP_PARALLEL_FOR_
        for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
            auto output_ptr = dst_origin + batch_idx * k_param_->ow * k_param_->oh * k_param_->oc_r4;
            for (int dz = 0; dz < k_param_->oc_r4; dz += 4) {
                auto dst_z    = output_ptr + dz * k_param_->ow * k_param_->oh;
                float *bias_z = reinterpret_cast<float *>(k_param_->bias) + dz;
                post_func_(dst_z, bias_z, k_param_->ow * k_param_->oh, 1);
            }
        }
    }
}

template void ArmConvLayerCommon::PostExec<float>(const std::vector<Blob *> &outputs);
template void ArmConvLayerCommon::PostExec<bfp16_t>(const std::vector<Blob *> &outputs);
}  // namespace TNN_NS
