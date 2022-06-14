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

#include "tnn/device/x86/acc/x86_reformat_layer_acc.h"

#include "tnn/device/x86/acc/compute/x86_compute_int8.h"
#include "tnn/device/x86/x86_common.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

Status X86ReformatLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(X86LayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);

    auto reformat_param = dynamic_cast<ReformatLayerParam *>(param);
    CHECK_PARAM_NULL(reformat_param);

    scale_buffer_.resize(inputs.size());
    if (reformat_param->src_type == DATA_TYPE_INT8 && reformat_param->dst_type == DATA_TYPE_FLOAT) {
        reformat_param->type = DEQUANT_ONLY;
        for (auto blob : outputs) {
            blob->GetBlobDesc().data_format = DATA_FORMAT_NCHW;
        }
    } else if (reformat_param->src_type == DATA_TYPE_FLOAT && reformat_param->dst_type == DATA_TYPE_INT8) {
        reformat_param->type = QUANT_ONLY;
        for (auto blob : outputs) {
            blob->GetBlobDesc().data_format = DATA_FORMAT_NHWC4;
        }
    } else {
        return Status(TNNERR_MODEL_ERR, "unsupport precision mode");
    }
    return allocateBufferParam(inputs, outputs);
}

X86ReformatLayerAcc::~X86ReformatLayerAcc() {}

Status X86ReformatLayerAcc::allocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ReformatLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

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

Status X86ReformatLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ReformatLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    for (int i = 0; i < inputs.size(); ++i) {
        auto dims   = outputs[i]->GetBlobDesc().dims;
        int batch   = dims[0];
        int channel = dims[1];
        int hw      = DimsVectorUtils::Count(dims, 2);
        if (param->type == DEQUANT_ONLY) {
            X86Int8ToFloat(handle_ptr<float *>(outputs[i]->GetHandle()),
                           handle_ptr<int8_t *>(inputs[i]->GetHandle()),
                           scale_buffer_[i].force_to<float *>(), batch, channel, hw);
        } else if (param->type == QUANT_ONLY) {
            X86FloatToInt8(handle_ptr<int8_t *>(outputs[i]->GetHandle()),
                           handle_ptr<float *>(inputs[i]->GetHandle()), scale_buffer_[i].force_to<float *>(),
                           batch, channel, hw);
        }
    }
    return TNN_OK;
}

REGISTER_X86_ACC(Reformat, LAYER_REFORMAT);

}  // namespace TNN_NS
