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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(Shape, LAYER_SHAPE);

Status ArmShapeLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    void *input_data  = GetBlobHandlePtr(inputs[0]->GetHandle());
    auto input_dims   = inputs[0]->GetBlobDesc().dims;
    void *output_data = GetBlobHandlePtr(outputs[0]->GetHandle());

    int *output_data_ptr = (int *)output_data; 
    for(int i = 0; i < input_dims.size(); i++) {
        output_data_ptr[i] = static_cast<int>(input_dims[i]);
    }
    output_data = reinterpret_cast<void *>(output_data_ptr);
    return TNN_OK;
}

REGISTER_ARM_ACC(Shape, LAYER_SHAPE);
REGISTER_ARM_LAYOUT(LAYER_SHAPE, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
