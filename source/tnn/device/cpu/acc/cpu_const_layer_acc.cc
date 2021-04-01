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
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(Const, LAYER_CONST);

Status CpuConstLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuConstLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_resource = dynamic_cast<ConstLayerResource *>(resource_);
    auto output_blob    = outputs[0];
    auto output_dims    = output_blob->GetBlobDesc().dims;
    auto count          = DimsVectorUtils::Count(output_dims);
    auto const_data_ptr = layer_resource->weight_handle.force_to<void *>();
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        memcpy(output_blob->GetHandle().base, const_data_ptr, count * sizeof(float));
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT32) {
        memcpy(output_blob->GetHandle().base, const_data_ptr, count * sizeof(int32_t));
    }
    return TNN_OK;
}

//REGISTER_CPU_ACC(Const, LAYER_CONST);
}  // namespace TNN_NS
