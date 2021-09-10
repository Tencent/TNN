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

#include "tnn/device/arm/acc/arm_reformat_layer_acc.h"

#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/half_utils_inner.h"

namespace TNN_NS {

Status ArmReformatLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);

    auto reformat_param = dynamic_cast<ReformatLayerParam *>(param);
    CHECK_PARAM_NULL(reformat_param);

    scale_buffer_.resize(inputs.size());

    if (reformat_param->src_format == reformat_param->dst_format) {
        if (reformat_param->src_type == DATA_TYPE_FLOAT && reformat_param->dst_type == DATA_TYPE_HALF) {
            reformat_param->type = NC4HW4FP32_2_NC8HW8FP16;
        } else if (reformat_param->src_type == DATA_TYPE_HALF && reformat_param->dst_type == DATA_TYPE_FLOAT) {
            reformat_param->type = NC8HW8FP16_2_NC4HW4FP32;
        } else if (reformat_param->src_type == DATA_TYPE_INT8 && reformat_param->dst_type == DATA_TYPE_FLOAT) {
            reformat_param->type = DEQUANT_ONLY;
        } else if (reformat_param->src_type == DATA_TYPE_FLOAT && reformat_param->dst_type == DATA_TYPE_INT8) {
            reformat_param->type = QUANT_ONLY;
        } else {
            if (reformat_param->src_type == DATA_TYPE_BFP16 || reformat_param->dst_type == DATA_TYPE_BFP16) {
                LOGE("unsupport precision mode, please dont use precision = low for int8");
            }
            return Status(TNNERR_MODEL_ERR, "unsupport precision mode");
        }
    } else if (reformat_param->src_format == DATA_FORMAT_NC4HW4 && reformat_param->dst_format == DATA_FORMAT_NCHW) {
        if (reformat_param->src_type == DATA_TYPE_FLOAT && reformat_param->dst_type == DATA_TYPE_FLOAT) {
            reformat_param->type = NC4HW4FP32_2_NCHWFP32;
        } else if (reformat_param->src_type == DATA_TYPE_HALF && reformat_param->dst_type == DATA_TYPE_HALF) {
            reformat_param->type = NC8HW8FP16_2_NCHWFP16;
        } else {
            LOGE("ArmReformatLayerAcc::Init Error: src_fmt: %d, dst_fmt: %d, src_type: %d, dst_type: %d\n",
                 reformat_param->src_format, reformat_param->dst_format, reformat_param->src_type,
                 reformat_param->dst_type);
            return Status(TNNERR_MODEL_ERR, "ArmReformatLayerAcc::Init unsupport reformat type");
        }
    } else if (reformat_param->src_format == DATA_FORMAT_NCHW && reformat_param->dst_format == DATA_FORMAT_NC4HW4) {
        if (reformat_param->src_type == DATA_TYPE_FLOAT && reformat_param->dst_type == DATA_TYPE_FLOAT) {
            reformat_param->type = NCHWFP32_2_NC4HW4FP32;
        } else if (reformat_param->src_type == DATA_TYPE_HALF && reformat_param->dst_type == DATA_TYPE_HALF) {
            reformat_param->type = NCHWFP16_2_NC8HW8FP16;
        } else {
            LOGE("ArmReformatLayerAcc::Init Error: src_fmt: %d, dst_fmt: %d, src_type: %d, dst_type: %d\n",
                 reformat_param->src_format, reformat_param->dst_format, reformat_param->src_type,
                 reformat_param->dst_type);
            return Status(TNNERR_MODEL_ERR, "ArmReformatLayerAcc::Init unsupport reformat type");
        }
    } else if ((reformat_param->src_format == DATA_FORMAT_NC4HW4 && reformat_param->dst_format == DATA_FORMAT_NHWC4) ||
               (reformat_param->src_format == DATA_FORMAT_NHWC4 && reformat_param->dst_format == DATA_FORMAT_NC4HW4)) {
        if (reformat_param->src_type == DATA_TYPE_INT8 && reformat_param->dst_type == DATA_TYPE_FLOAT) {
            reformat_param->type = DEQUANT_ONLY;
        } else if (reformat_param->src_type == DATA_TYPE_FLOAT && reformat_param->dst_type == DATA_TYPE_INT8) {
            reformat_param->type = QUANT_ONLY;
        } else {
            LOGE("ArmReformatLayerAcc::Init Error: src_fmt: %d, dst_fmt: %d, src_type: %d, dst_type: %d\n",
                 reformat_param->src_format, reformat_param->dst_format, reformat_param->src_type,
                 reformat_param->dst_type);
            return Status(TNNERR_MODEL_ERR, "ArmReformatLayerAcc::Init unsupport reformat type");
        }
    } else {
        LOGE("ArmReformatLayerAcc::Init Error: src_fmt: %d, dst_fmt: %d, src_type: %d, dst_type: %d\n",
             reformat_param->src_format, reformat_param->dst_format, reformat_param->src_type,
             reformat_param->dst_type);
        return Status(TNNERR_MODEL_ERR, "ArmReformatLayerAcc::Init unsupport reformat type");
    }
    return allocateBufferParam(inputs, outputs);
}

ArmReformatLayerAcc::~ArmReformatLayerAcc() {}

Status ArmReformatLayerAcc::allocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ReformatLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    if (param->type == NC4HW4FP32_2_NC8HW8FP16 || param->type == NC8HW8FP16_2_NC4HW4FP32) {
        return TNN_OK;
    }

    for (int idx = 0; idx < inputs.size(); ++idx) {
        if (param->src_type != param->dst_type && !scale_buffer_[idx].GetBytesSize()) {
            auto dims_output    = outputs[idx]->GetBlobDesc().dims;
            int total_byte_size = ROUND_UP(dims_output[1], 4) * sizeof(float);
            IntScaleResource *reformat_scale;
            if (param->src_type == DATA_TYPE_INT8) {
                reformat_scale = reinterpret_cast<BlobInt8 *>(inputs[idx])->GetIntResource();
            } else {
                reformat_scale = reinterpret_cast<BlobInt8 *>(outputs[idx])->GetIntResource();
            }
            const float *scale = reformat_scale->scale_handle.force_to<float *>();
            int scale_cnt      = reformat_scale->scale_handle.GetDataCount();
            RawBuffer temp_buffer(total_byte_size);
            float *temp_ptr = temp_buffer.force_to<float *>();
            for (int i = 0; i < dims_output[1]; i++) {
                int scale_idx = scale_cnt == 1 ? 0 : i;
                if (param->type == QUANT_ONLY)
                    temp_ptr[i] = 1.0 / scale[scale_idx];
                if (param->type == DEQUANT_ONLY)
                    temp_ptr[i] = scale[scale_idx];
            }
            scale_buffer_[idx] = temp_buffer;
        }
    }

    return TNN_OK;
}

Status ArmReformatLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ReformatLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    for (int i = 0; i < inputs.size(); ++i) {
        auto dims   = outputs[i]->GetBlobDesc().dims;
        int batch   = DimsFunctionUtils::GetDim(dims, 0);
        int channel = DimsFunctionUtils::GetDim(dims, 1);
        int hw      = DimsVectorUtils::Count(dims, 2);
        if (param->type == DEQUANT_ONLY) {
            Int8ToFloat(reinterpret_cast<float *>(GetBlobHandlePtr(outputs[i]->GetHandle())),
                        reinterpret_cast<int8_t *>(GetBlobHandlePtr(inputs[i]->GetHandle())),
                        scale_buffer_[i].force_to<float *>(), batch, channel, hw);
        } else if (param->type == QUANT_ONLY) {
            FloatToInt8(reinterpret_cast<int8_t *>(GetBlobHandlePtr(outputs[i]->GetHandle())),
                        reinterpret_cast<float *>(GetBlobHandlePtr(inputs[i]->GetHandle())),
                        scale_buffer_[i].force_to<float *>(), batch, channel, hw);
        } else if (param->type == NC4HW4FP32_2_NCHWFP32) {
            auto dst_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[i]->GetHandle()));
            auto src_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(inputs[i]->GetHandle()));
            UnpackFloatBlob(dst_ptr, src_ptr, batch, channel, hw);
        } else if (param->type == NCHWFP32_2_NC4HW4FP32) {
            auto dst_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(outputs[i]->GetHandle()));
            auto src_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(inputs[i]->GetHandle()));
            PackFloatBlob(dst_ptr, src_ptr, batch, channel, hw);
        }
#if TNN_ARM82
        else if (param->type == NC4HW4FP32_2_NC8HW8FP16) {
            FloatC4ToHalfC8(reinterpret_cast<fp16_t *>(GetBlobHandlePtr(outputs[i]->GetHandle())),
                            reinterpret_cast<float *>(GetBlobHandlePtr(inputs[i]->GetHandle())), batch, channel, hw);
        } else if (param->type == NC8HW8FP16_2_NC4HW4FP32) {
            HalfC8ToFloatC4(reinterpret_cast<float *>(GetBlobHandlePtr(outputs[i]->GetHandle())),
                            reinterpret_cast<fp16_t *>(GetBlobHandlePtr(inputs[i]->GetHandle())), batch, channel, hw);
        } else if (param->type == NC8HW8FP16_2_NCHWFP16) {
            auto dst_ptr = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(outputs[i]->GetHandle()));
            auto src_ptr = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(inputs[i]->GetHandle()));
            UnpackHalfBlob(dst_ptr, src_ptr, batch, channel, hw);
        } else if (param->type == NCHWFP16_2_NC8HW8FP16) {
            auto dst_ptr = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(outputs[i]->GetHandle()));
            auto src_ptr = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(inputs[i]->GetHandle()));
            PackHalfBlob(dst_ptr, src_ptr, batch, channel, hw);
        }
#endif  // TNN_ARM82
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(Reformat, LAYER_REFORMAT)
REGISTER_ARM_LAYOUT(LAYER_REFORMAT, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
