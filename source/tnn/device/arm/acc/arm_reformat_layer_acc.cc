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

namespace TNN_NS {

Status ArmReformatLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);

    auto reformat_param = dynamic_cast<ReformatLayerParam *>(param);
    CHECK_PARAM_NULL(reformat_param);

    if (reformat_param->src_type == DATA_TYPE_INT8 && reformat_param->dst_type == DATA_TYPE_FLOAT) {
        reformat_param->type = DEQUANT_ONLY;
        for (auto blob : outputs) {
            blob->GetBlobDesc().data_format = DATA_FORMAT_NC4HW4;
        }
    } else if (reformat_param->src_type == DATA_TYPE_FLOAT && reformat_param->dst_type == DATA_TYPE_INT8) {
        reformat_param->type = QUANT_ONLY;
        for (auto blob : outputs) {
            blob->GetBlobDesc().data_format = DATA_FORMAT_NHWC4;
        }
    } else {
        if (reformat_param->src_type == DATA_TYPE_BFP16 || reformat_param->dst_type == DATA_TYPE_BFP16) {
            LOGE("unsupport precision mode, please dont use precision = low for int8");
        }
        return Status(TNNERR_MODEL_ERR, "unsupport precision mode");
    }
    return allocateBufferParam(inputs, outputs);
}

ArmReformatLayerAcc::~ArmReformatLayerAcc() {}

Status ArmReformatLayerAcc::allocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ReformatLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    if (param->src_type != param->dst_type && !scale_buffer_.GetBytesSize()) {
        auto dims_output    = outputs[0]->GetBlobDesc().dims;
        int total_byte_size = ROUND_UP(dims_output[1], 4) * sizeof(float);
        IntScaleResource *reformat_scale;
        if (param->src_type == DATA_TYPE_INT8) {
            reformat_scale = reinterpret_cast<BlobInt8 *>(inputs[0])->GetIntResource();
        } else {
            reformat_scale = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource();
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
        scale_buffer_ = temp_buffer;
    }

    return TNN_OK;
}

Status ArmReformatLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto dims = outputs[0]->GetBlobDesc().dims;

    auto param = dynamic_cast<ReformatLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    if (param->type == DEQUANT_ONLY) {
        Int8ToFloat(reinterpret_cast<float *>(GetBlobHandlePtr(outputs[0]->GetHandle())),
                    reinterpret_cast<int8_t *>(GetBlobHandlePtr(inputs[0]->GetHandle())), 
                    scale_buffer_.force_to<float *>(), dims[0], dims[1], dims[2] * dims[3]);
    } else if (param->type == QUANT_ONLY) {
        FloatToInt8(reinterpret_cast<int8_t *>(GetBlobHandlePtr(outputs[0]->GetHandle())),
                    reinterpret_cast<float *>(GetBlobHandlePtr(inputs[0]->GetHandle())), 
                    scale_buffer_.force_to<float *>(), dims[0], dims[1], dims[2] * dims[3]);
    }
    return TNN_OK;
}

ArmTypeLayerAccRegister<TypeLayerAccCreator<ArmReformatLayerAcc>> g_arm_reformat_layer_acc_register(LAYER_REFORMAT);

}  // namespace TNN_NS
