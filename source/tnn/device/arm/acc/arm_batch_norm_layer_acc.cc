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

#include "tnn/device/arm/acc/arm_batch_norm_layer_acc.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils_inner.h"

namespace TNN_NS {

ArmBatchNormLayerAcc::~ArmBatchNormLayerAcc() {}

Status ArmBatchNormLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    return allocateBufferParam(inputs, outputs);
}

Status ArmBatchNormLayerAcc::allocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    BatchNormLayerResource *batch_norm_res = dynamic_cast<BatchNormLayerResource *>(resource_);
    CHECK_PARAM_NULL(batch_norm_res);

    RawBuffer scale_handle = batch_norm_res->scale_handle;
    RawBuffer bias_handle  = batch_norm_res->bias_handle;

    if (scale_handle.GetDataType() == DATA_TYPE_HALF)
        scale_handle = ConvertHalfHandle(scale_handle);
    if (bias_handle.GetDataType() == DATA_TYPE_HALF)
        bias_handle = ConvertHalfHandle(bias_handle);

    auto data_bytes_size = DataTypeUtils::GetBytesSize(scale_handle.GetDataType());

    shared_channel_ = (scale_handle.GetBytesSize() == DataTypeUtils::GetBytesSize(scale_handle.GetDataType()));

    if (!buffer_scale_.GetBytesSize()) {
        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            int channel       = shared_channel_ ? 1 : dims_output[1];
            int channel_count = shared_channel_ ? 1 : ROUND_UP(dims_output[1], 8);
            RawBuffer temp_buffer(channel_count * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
            Float2Half(temp_buffer.force_to<fp16_t *>(), scale_handle.force_to<float *>(), channel);
            buffer_scale_ = temp_buffer;
        } else {
            int channel       = shared_channel_ ? 1 : dims_output[1];
            int channel_count = shared_channel_ ? 1 : ROUND_UP(dims_output[1], 4);
            RawBuffer temp_buffer(channel_count * data_bytes_size);
            memcpy(temp_buffer.force_to<void *>(), scale_handle.force_to<void *>(), channel * data_bytes_size);
            buffer_scale_ = temp_buffer;
        }
    }

    if (!buffer_bias_.GetBytesSize()) {
        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            int channel       = shared_channel_ ? 1 : dims_output[1];
            int channel_count = shared_channel_ ? 1 : ROUND_UP(dims_output[1], 8);
            RawBuffer temp_buffer(channel_count * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
            if (bias_handle.force_to<void *>()) {
                Float2Half(temp_buffer.force_to<fp16_t *>(), bias_handle.force_to<float *>(), channel);
            }
            buffer_bias_ = temp_buffer;
        } else {
            int channel       = shared_channel_ ? 1 : dims_output[1];
            int channel_count = shared_channel_ ? 1 : ROUND_UP(dims_output[1], 4);
            RawBuffer temp_buffer(channel_count * data_bytes_size);
            if (bias_handle.force_to<void *>()) {
                memcpy(temp_buffer.force_to<void *>(), bias_handle.force_to<void *>(), channel * data_bytes_size);
            }
            buffer_bias_ = temp_buffer;
        }
    }

    return TNN_OK;
}

template <typename T>
Status ArmBatchNormLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input       = inputs[0];
    auto output      = outputs[0];
    auto dims_input  = input->GetBlobDesc().dims;
    auto dims_output = output->GetBlobDesc().dims;

    auto ic = dims_input[1], input_slice = UP_DIV(dims_input[1], 4);
    auto oc = dims_output[1], output_slice = UP_DIV(dims_output[1], 4);
    auto i_hw = DimsVectorUtils::Count(dims_input, 2);
    auto o_hw = DimsVectorUtils::Count(dims_output, 2);

    auto batch = dims_output[0];

    T *input_orign  = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));
    T *output_orign = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    float *k_data = buffer_scale_.force_to<float *>();
    float *b_data = buffer_bias_.force_to<float *>();

    auto src_z_step = i_hw * 4;
    auto dst_z_step = o_hw * 4;

    for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
        auto input_ptr  = input_orign + batch_idx * input_slice * 4 * i_hw;
        auto output_ptr = output_orign + batch_idx * output_slice * 4 * o_hw;

        if (!shared_channel_) {
            for (int dz = 0; dz < output_slice; dz++) {
                for (int x_i = 0; x_i < o_hw; x_i++) {
                    Float4 input_v  = Float4::load(input_ptr + dz * src_z_step + x_i * 4);
                    Float4 k_data_v = Float4::load(k_data + dz * 4);
                    Float4 b_data_v = Float4::load(b_data + dz * 4);
                    Float4::mla(b_data_v, input_v, k_data_v);
                    Float4::save(output_ptr + dz * dst_z_step + x_i * 4, b_data_v);
                }
            }
        } else {
            Float4 k_data_v = Float4(k_data[0]);
            Float4 b_data_v = Float4(b_data[0]);
            for (int dz = 0; dz < output_slice; dz++) {
                for (int x_i = 0; x_i < o_hw; x_i++) {
                    Float4 input_v = Float4::load(input_ptr + dz * src_z_step + x_i * 4);
                    Float4 dst_v   = b_data_v;
                    Float4::mla(dst_v, input_v, k_data_v);
                    Float4::save(output_ptr + dz * dst_z_step + x_i * 4, dst_v);
                }
            }
        }
    }

    return TNN_OK;
}

Status ArmBatchNormLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto in_data_type = inputs[0]->GetBlobDesc().data_type;
    if (in_data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (in_data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    }
#if TNN_ARM82
    else if (in_data_type == DATA_TYPE_HALF) {
        return ExecFp16(inputs, outputs);
    }
#endif
    else {
        return TNNERR_LAYER_ERR;
    }
}

REGISTER_ARM_ACC(BatchNorm, LAYER_BATCH_NORM)
REGISTER_ARM_PRECISION_FP16(LAYER_BATCH_NORM)
REGISTER_ARM_LAYOUT(LAYER_BATCH_NORM, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
