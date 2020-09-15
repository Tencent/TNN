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

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(PixelShuffle, LAYER_PIXEL_SHUFFLE);

Status ArmPixelShuffleLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto in_data_type   = inputs[0]->GetBlobDesc().data_type;
    if (in_data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (in_data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    } else {
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }
}

template<typename T>
static Status ExecFactor1(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_dims  = inputs[0]->GetBlobDesc().dims;

    auto *input_ptr  = static_cast<T *>(inputs[0]->GetHandle().base);
    auto *output_ptr = static_cast<T *>(outputs[0]->GetHandle().base);

    int data_byte_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);
    auto size_in_bytes = input_dims[0] * ROUND_UP(input_dims[1], 4) * input_dims[2] * input_dims[3] * data_byte_size;

    memcpy(output_ptr, input_ptr, size_in_bytes);

    return TNN_OK;
}

template <typename T>
Status ArmPixelShuffleLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param         = dynamic_cast<PixelShuffleLayerParam *>(param_);
    int upscale_factor = param->upscale_factor;

    if (upscale_factor == 1) {
        return ExecFactor1<T>(inputs, outputs);
    } else {
        return Status(TNNERR_PARAM_ERR, "pixel shuffle upscale factor not support");
    }
}

REGISTER_ARM_ACC(PixelShuffle, LAYER_PIXEL_SHUFFLE);

}  // namespace TNN_NS
