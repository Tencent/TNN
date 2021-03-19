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

#include "cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
namespace TNN_NS {

DECLARE_CPU_ACC(Shape, LAYER_SHAPE);

Status CpuShapeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuShapeLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    const auto& input_blob = inputs[0];
    const auto& input_dims = inputs[0]->GetBlobDesc().dims;
    const auto& output_blob = outputs[0];
    const auto& data_type = output_blob->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_INT32) {
        auto output_data = static_cast<int32_t*>(output_blob->GetHandle().base);
        for (int i = 0; i < input_dims.size(); ++i) {
            output_data[i] = input_dims[i];
        }
    } else {
        LOGE("output blob of Shape Layer has wrong data type \n");
        return Status(TNNERR_COMMON_ERROR, "output blob has wrong data type");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Shape, LAYER_SHAPE);
}  // namespace TNN_NS
