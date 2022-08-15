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
#include "tnn/core/blob_int8.h"
#include "tnn/utils/naive_compute.h"
namespace TNN_NS {

DECLARE_CPU_BINARY_OP_ACC(Add, LAYER_ADD);

Status CpuAddLayerAcc::Calculate(const std::vector<Blob *> &input_blobs, const std::vector<void *> &input_ptrs,
                                 const std::vector<DimsVector> &input_shapes, Blob *output) {
    if (output->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        CPU_ADD(input_ptrs, input_shapes, output->GetHandle().base, output->GetBlobDesc().dims);
    } else if (output->GetBlobDesc().data_type == DATA_TYPE_INT32) {
        void *output_data = output->GetHandle().base;
        const auto &output_dims = output->GetBlobDesc().dims;
        CPU_ELEMENT_WISE<int, int>(input_ptrs, input_shapes, output_data, output_dims,
                                  [](int a, int b) -> int { return a + b; });
    } else if (output->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        std::vector<float *> scale_ptrs;
        std::vector<int8_t *> zero_point_ptrs;

        for (size_t inid = 0; inid < input_blobs.size(); inid++) {
            scale_ptrs.push_back(
                reinterpret_cast<BlobInt8 *>(input_blobs[inid])->GetIntResource()->scale_handle.force_to<float *>());            
            zero_point_ptrs.push_back(
                reinterpret_cast<BlobInt8 *>(input_blobs[inid])->GetIntResource()->zero_point_handle.force_to<int8_t *>());
        }
        CPU_ADD_BIAS(input_ptrs, scale_ptrs, zero_point_ptrs,
                reinterpret_cast<BlobInt8 *>(input_blobs[0])->GetIntResource()->scale_handle.GetDataCount(),
                output->GetHandle().base,
                reinterpret_cast<BlobInt8 *>(output)->GetIntResource()->scale_handle.force_to<float *>(),
                reinterpret_cast<BlobInt8 *>(output)->GetIntResource()->zero_point_handle.force_to<int8_t *>(),
                output->GetBlobDesc().dims);        
    } else {
        LOGE("Error: CpuAddLayerAcc don't support data type: %d\n", output->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuAddLayerAcc don't support data type");
    }
    return TNN_OK;
}
REGISTER_CPU_ACC(Add, LAYER_ADD);

}  // namespace TNN_NS
