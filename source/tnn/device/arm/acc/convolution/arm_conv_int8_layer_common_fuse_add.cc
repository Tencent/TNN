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

#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_common_fuse_add.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {
/*
ArmConvInt8LayerCommon as the last conv int8 solution
*/
bool ArmConvInt8LayerCommonFuseAdd::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs) {
    if (ArmConvInt8LayerCommon::isPrefered(param, inputs, outputs) && param->fusion_type != FusionType_None) {
        return true;
    }
    return false;
}

ArmConvInt8LayerCommonFuseAdd::~ArmConvInt8LayerCommonFuseAdd() {}

Status ArmConvInt8LayerCommonFuseAdd::allocateBufferAddScale(const std::vector<Blob *> &inputs,
                                                             const std::vector<Blob *> &outputs) {
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    // alloc add scale buffer
    if (!buffer_add_scale_.GetBytesSize()) {
        auto dims_output = outputs[0]->GetBlobDesc().dims;
        int total_byte_size =
            ROUND_UP(dims_output[1], 4) * DataTypeUtils::GetBytesSize(conv_res->scale_handle.GetDataType());

        const float *i_scale =
            reinterpret_cast<BlobInt8 *>(inputs[1])->GetIntResource()->scale_handle.force_to<float *>();

        const float *o_scale =
            reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.force_to<float *>();

        int scale_len_i = reinterpret_cast<BlobInt8 *>(inputs[1])->GetIntResource()->scale_handle.GetDataCount();
        int scale_len_o = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.GetDataCount();
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

    return TNN_OK;
}


Status ArmConvInt8LayerCommonFuseAdd::Init(Context *context, LayerParam *param, LayerResource *resource,
                                           const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmConvInt8LayerCommon::Init(context, param, resource, inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferAddScale(inputs, outputs), TNN_OK);

    int max_num_threads = OMP_CORES_;
    if (!buffer_add_tmpin_.GetBytesSize()) {
        const int oc_round4   = ROUND_UP(outputs[0]->GetBlobDesc().dims[1], 4);
        const int buffer_size = oc_round4 * NEON_INT8CONV_TILE_HW * max_num_threads;

        RawBuffer temp_buffer(buffer_size);
        memset(temp_buffer.force_to<void *>(), 0, buffer_size);
        buffer_add_tmpin_ = temp_buffer;
    }

    return TNN_OK;
}

Status ArmConvInt8LayerCommonFuseAdd::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param);
    auto input  = inputs[0];
    auto output = outputs[0];

    DataType data_type = output->GetBlobDesc().data_type;
    int data_byte_size = DataTypeUtils::GetBytesSize(data_type);

    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;
    const int batch  = dims_output[0];
    auto ic          = dims_input[1];
    auto ic_calc     = ic < 4 ? ic : k_param_->ic_r4;

    int8_t *input_data  = reinterpret_cast<int8_t *>(GetBlobHandlePtr(input->GetHandle()));
    int8_t *add_input   = reinterpret_cast<int8_t *>(GetBlobHandlePtr(inputs[1]->GetHandle()));
    int8_t *output_data = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));

    const int crs_div8   = UP_DIV(ic_calc * conv_param->kernels[1] * conv_param->kernels[0], 8);
    const int tile_count = UP_DIV(k_param_->oh * k_param_->ow, NEON_INT8CONV_TILE_HW);
    for (int n = 0; n < batch; ++n) {
        const auto input_batch = input_data + n * k_param_->iw * k_param_->ih * k_param_->ic_r4;
        auto output_batch      = output_data + n * k_param_->ow * k_param_->oh * k_param_->oc_r4;
        auto add_input_batch   = add_input + n * k_param_->ow * k_param_->oh * k_param_->oc_r4;

        OMP_PARALLEL_FOR_GUIDED_
        for (int t_idx = 0; t_idx < tile_count; t_idx++) {
            int thread_id          = OMP_TID_;
            int8_t *input_kernel   = nullptr;
            const int hw_start     = t_idx * NEON_INT8CONV_TILE_HW;
            const int real_hw_tile = MIN(k_param_->oh * k_param_->ow - hw_start, NEON_INT8CONV_TILE_HW);
            auto gemm_work_space   = buffer_gemm_work_space_.force_to<int8_t *>();
            // im2col
            if (im_col_func_) {
                input_kernel = buffer_im2col_.force_to<int8_t *>() + crs_div8 * NEON_INT8CONV_TILE_HW * 8 * thread_id;
                im_col_func_(input_kernel, input_batch, conv_param, hw_start, real_hw_tile, crs_div8, k_param_.get());
            } else {
                input_kernel = input_batch + hw_start * ic_calc;
            }
            auto output_kernel    = output_batch + hw_start * k_param_->oc_r4;
            auto add_input_kernel = add_input_batch + hw_start * k_param_->oc_r4;
            // gemm int8
            if (real_hw_tile == NEON_INT8CONV_TILE_HW) {
                GemmAddInt8(output_kernel, input_kernel, gemm_work_space, reinterpret_cast<int8_t *>(k_param_->fil_ptr),
                            reinterpret_cast<int32_t *>(k_param_->bias), k_param_->scale, crs_div8, crs_div8 * 8,
                            k_param_->oc_r4, add_input_kernel, buffer_add_scale_.force_to<float *>());
            } else {
                int8_t *outptr_tmp =
                    buffer_tmpout_.force_to<int8_t *>() + k_param_->oc_r4 * NEON_INT8CONV_TILE_HW * thread_id;
                int8_t *add_input_ptr_tmp =
                    buffer_add_tmpin_.force_to<int8_t *>() + k_param_->oc_r4 * NEON_INT8CONV_TILE_HW * thread_id;
                memcpy(add_input_ptr_tmp, add_input_kernel, real_hw_tile * k_param_->oc_r4);
                GemmAddInt8(outptr_tmp, input_kernel, gemm_work_space, reinterpret_cast<int8_t *>(k_param_->fil_ptr),
                            reinterpret_cast<int32_t *>(k_param_->bias), k_param_->scale, crs_div8, crs_div8 * 8,
                            k_param_->oc_r4, add_input_ptr_tmp, buffer_add_scale_.force_to<float *>());
                memcpy(output_kernel, outptr_tmp, real_hw_tile * k_param_->oc_r4);
            }
        }
        // only support relu activation
        if (conv_param->activation_type == ActivationType_ReLU) {
            ReluInt8(output_batch, output_batch, k_param_->ow * k_param_->oh * k_param_->oc_r4);
        }
    }
    return TNN_OK;
}

}  // namespace TNN_NS
