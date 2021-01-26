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

DECLARE_CPU_ACC(Size, LAYER_SIZE);

Status CpuSizeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuSizeLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    void *output_data = ((char *)outputs[0]->GetHandle().base) + outputs[0]->GetHandle().bytes_offset;
    auto output_data_type = outputs[0]->GetBlobDesc().data_type;

    const int ele_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);

    const int count = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims);
    if (output_data_type == DATA_TYPE_INT32) {
        *((int *)output_data) = count;
    } else {
        LOGE("unsupport data type to cast\n");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Size, LAYER_SIZE);
}  // namespace TNN_NS
