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

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/acc/x86_batch_norm_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "tnn/interpreter/layer_resource_generator.h"

namespace TNN_NS {

X86BatchNormLayerAcc::~X86BatchNormLayerAcc() {}

Status X86BatchNormLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto res = dynamic_cast<BatchNormLayerResource *>(resource);
    CHECK_PARAM_NULL(res);

    Status ret;
    if (res->scale_handle.GetDataType() == DATA_TYPE_HALF) {
        LayerResource *fp32_res = nullptr;
        RETURN_ON_NEQ(ConvertHalfResource(LAYER_BATCH_NORM, res, &fp32_res), TNN_OK);
        bn_acc_f32_resource_ = std::shared_ptr<LayerResource>(fp32_res);
        ret = X86LayerAcc::Init(context, param, bn_acc_f32_resource_.get(), inputs, outputs);
    } else {
        ret = X86LayerAcc::Init(context, param, resource, inputs, outputs);
    }

    RETURN_ON_NEQ(ret, TNN_OK);
    return TNN_OK;
}

Status X86BatchNormLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    
    auto resource = dynamic_cast<BatchNormLayerResource *>(resource_);
    if (!resource) {
        return Status(TNNERR_MODEL_ERR, "Error: BatchNormLayerResource is nil");
    }

    auto input_blob        = inputs[0];
    auto output_blob       = outputs[0];

    auto x86_fma_func = X86_FMA<Float4, 4>;
    if (arch_ == avx2) {
        x86_fma_func = X86_FMA<Float8, 8>;
    }

    RawBuffer scale_handle = resource->scale_handle;
    bool shared_channel     = scale_handle.GetBytesSize() == DataTypeUtils::GetBytesSize(scale_handle.GetDataType());
    RawBuffer bias_handle  = resource->bias_handle;
    bool has_bias          = bias_handle.GetDataCount() > 0; 

    x86_fma_func(handle_ptr<float *>(input_blob->GetHandle()),
            handle_ptr<float *>(output_blob->GetHandle()),
            scale_handle.force_to<float *>(), bias_handle.force_to<float *>(),
            shared_channel, has_bias, output_blob->GetBlobDesc().dims);

    return TNN_OK;
}

REGISTER_X86_ACC(BatchNorm, LAYER_BATCH_NORM);

}  // namespace TNN_NS
