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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"
#include "immintrin.h"

namespace TNN_NS {

DECLARE_X86_ACC(Scale, LAYER_SCALE);

Status X86ScaleLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    
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

REGISTER_X86_ACC(Scale, LAYER_SCALE);

}  // namespace TNN_NS
