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

#include "tnn/device/arm/acc/arm_prelu_layer_acc.h"

#include <cmath>
#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/half_utils_inner.h"

namespace TNN_NS {

ArmPReluLayerAcc::~ArmPReluLayerAcc(){};

Status ArmPReluLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                              const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    return allocateBufferParam(inputs, outputs);
}

Status ArmPReluLayerAcc::allocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    auto layer_param = dynamic_cast<PReluLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    PReluLayerResource *prelu_res = dynamic_cast<PReluLayerResource *>(resource_);
    CHECK_PARAM_NULL(prelu_res);

    RawBuffer slope_handle = prelu_res->slope_handle;

    if (slope_handle.GetDataType() == DATA_TYPE_HALF)
        slope_handle = ConvertHalfHandle(slope_handle);

    auto data_bytes_size = DataTypeUtils::GetBytesSize(slope_handle.GetDataType());

    if (!buffer_slope_.GetBytesSize()) {
        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            int channel       = layer_param->channel_shared ? 1 : dims_output[1];
            int channel_count = layer_param->channel_shared ? 1 : ROUND_UP(dims_output[1], 8);
            RawBuffer temp_buffer(channel_count * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF));
            Float2Half(temp_buffer.force_to<fp16_t *>(), slope_handle.force_to<float *>(), channel);
            buffer_slope_ = temp_buffer;
        } else {
            int channel       = layer_param->channel_shared ? 1 : dims_output[1];
            int channel_count = layer_param->channel_shared ? 1 : ROUND_UP(dims_output[1], 4);
            RawBuffer temp_buffer(channel_count * data_bytes_size);
            memcpy(temp_buffer.force_to<void *>(), slope_handle.force_to<void *>(), channel * data_bytes_size);
            buffer_slope_ = temp_buffer;
        }
    }

    return TNN_OK;
}

template <typename T>
Status ArmPReluLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PReluLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    auto dims              = inputs[0]->GetBlobDesc().dims;
    const int channel      = dims[1];
    const int hw           = DimsVectorUtils::Count(dims, 2);
    const int count        = dims[0] * ROUND_UP(dims[1], 4) * hw;

    const float *slope_data = buffer_slope_.force_to<float *>();

    T *input_data  = reinterpret_cast<T *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    T *output_data = reinterpret_cast<T *>(GetBlobHandlePtr(outputs[0]->GetHandle()));
    if (layer_param->channel_shared) {
        for (int n = 0; n < UP_DIV(count, 4); n++) {
            Float4 v_data = Float4::load(input_data + n * 4);
            Float4 v_res  = Float4::bsl_clt(v_data, Float4(0.f), v_data * slope_data[0], v_data);
            Float4::save(output_data + n * 4, v_res);
        }
    } else {
        for (int batch_idx = 0; batch_idx < dims[0]; ++batch_idx) {
            auto input_ptr  = input_data + batch_idx * hw * ROUND_UP(channel, 4);
            auto output_ptr = output_data + batch_idx * hw * ROUND_UP(channel, 4);
            for (int dz = 0; dz < UP_DIV(channel, 4); ++dz) {
                T *src_z       = input_ptr + dz * hw * 4;
                T *dst_z       = output_ptr + dz * hw * 4;
                Float4 v_slope = Float4::load(slope_data + dz * 4);
                for (int p = 0; p < hw; p++) {
                    Float4 v_data = Float4::load(src_z + p * 4);
                    Float4 v_res  = Float4::bsl_clt(v_data, Float4(0.f), v_data * v_slope, v_data);
                    Float4::save(dst_z + p * 4, v_res);
                }
            }
        }
    }

    return TNN_OK;
}


Status ArmPReluLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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

REGISTER_ARM_ACC(PRelu, LAYER_PRELU)
REGISTER_ARM_PRECISION_FP16(LAYER_PRELU)
REGISTER_ARM_LAYOUT(LAYER_PRELU, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
