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

#include "cpu_binary_op_layer_acc.h"
#include "tnn/utils/naive_compute.h"
namespace TNN_NS {

DECLARE_CPU_BINARY_OP_ACC(Max, LAYER_MAXIMUM);

Status CpuMaxLayerAcc::Calculate(const std::vector<Blob *> &input_blobs, const std::vector<void *> &input_ptrs,
                                 const std::vector<DimsVector> &input_shapes, Blob *output) {
    if (output->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        CPU_MAX(input_ptrs, input_shapes, output->GetHandle().base, output->GetBlobDesc().dims);
    } else {
        LOGE("Error: CpuMaxLayerAcc don't support data type: %d\n", output->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuMaxLayerAcc don't support data type");
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(Max, LAYER_MAXIMUM);

}  // namespace TNN_NS
