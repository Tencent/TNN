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

#include "tnn/device/arm/acc/Half8.h"
#include "tnn/device/arm/acc/arm_relu_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

#if TNN_ARM82
Status ArmReluLayerAcc::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto dims   = output->GetBlobDesc().dims;

    auto count  = dims[0] * ROUND_UP(dims[1], 8) * DimsVectorUtils::Count(dims, 2);
    fp16_t *dst = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));
    fp16_t *src = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(input->GetHandle()));
    Half8 vzero = Half8((fp16_t)0.f);
    for (long i = 0; i < count; i += 8) {
        Half8::save(dst + i, Half8::max(Half8::load(src + i), vzero));
    }

    return TNN_OK;
}
#endif

}  // namespace TNN_NS
