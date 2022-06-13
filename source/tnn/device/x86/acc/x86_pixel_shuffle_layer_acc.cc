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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_X86_ACC(PixelShuffle, LAYER_PIXEL_SHUFFLE);

Status X86PixelShuffleLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param   = dynamic_cast<PixelShuffleLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    int upscale_factor = layer_param->upscale_factor;
    auto input_blob    = inputs[0];
    auto input_dims    = input_blob->GetBlobDesc().dims;
    auto output_blob   = outputs[0];
    auto output_dims   = output_blob->GetBlobDesc().dims;
    int slice_size     = DimsVectorUtils::Count(output_dims, 0, 2);
    auto input_h       = input_dims[2];
    auto input_w       = input_dims[3];
    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto input_prt  = handle_ptr<float *>(input_blob->GetHandle());
        auto output_ptr = handle_ptr<float *>(output_blob->GetHandle());
        for (int s = 0; s < slice_size; ++s) {
            for (int i = 0; i < upscale_factor; ++i) {
                for (int j = 0; j < upscale_factor; ++j) {
                    for (int h = 0; h < input_h; ++h) {
                        for (int w = 0; w < input_w; ++w) {
                            output_ptr[s * input_h * upscale_factor * input_w * upscale_factor +
                                       h * upscale_factor * input_w * upscale_factor + i * input_w * upscale_factor +
                                       w * upscale_factor + j] =
                                input_prt[s * upscale_factor * upscale_factor * input_h * input_w +
                                          i * upscale_factor * input_h * input_w + j * input_h * input_w + h * input_w +
                                          w];
                        }
                    }
                }
            }
        }
    }
    return TNN_OK;
}

REGISTER_X86_ACC(PixelShuffle, LAYER_PIXEL_SHUFFLE);
}  // namespace TNN_NS
