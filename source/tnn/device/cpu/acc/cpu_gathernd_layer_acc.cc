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

DECLARE_CPU_ACC(GatherND, LAYER_GATHERND);

Status CpuGatherNDLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuGatherNDLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<GatherNDLayerParam*>(param_);
    CHECK_PARAM_NULL(layer_param);
    int batch_dims = layer_param->batch_dims;

    if (batch_dims != 0) {
        return Status(TNNERR_PARAM_ERR, "GatherNDLayerParam has invalid param batch_dims");
    }

    auto input_data_dims = (*(inputs.begin()))->GetBlobDesc().dims;
    auto input_data_ptr = (char*)(*(inputs.begin()))->GetHandle().base + (*(inputs.begin()))->GetHandle().bytes_offset;
    auto output_data_ptr = (char*)(*(outputs.begin()))->GetHandle().base + (*(outputs.begin()))->GetHandle().bytes_offset;
    auto input_stride = DimsFunctionUtils::StrideOfShape(input_data_dims);

    auto indices_dims = (*(inputs.rbegin()))->GetBlobDesc().dims;
    int *indices_data_ptr = (int *)(*(inputs.rbegin()))->GetHandle().base;

    if (indices_dims[indices_dims.size() - 1] > input_data_dims.size()) {
        return Status(TNNERR_PARAM_ERR, "GatherNDLayerParam has invalid param indices_dims");
    }
    
    const int slice_index_size = indices_dims[indices_dims.size()-1];
    const int ele_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);
    const int ele_count =
        DimsVectorUtils::Count(input_data_dims, input_data_dims.size() - indices_dims[indices_dims.size() - 1], -1);
    const int output_slice_count = DimsVectorUtils::Count(indices_dims, 0, (int)indices_dims.size()-1);
    for (int i=0; i<output_slice_count; i++) {
        auto output_index = i;

        int *indices_ptr = indices_data_ptr + i * slice_index_size;

        int input_index = 0;
        for (int ii=0; ii<slice_index_size; ii++) {
            input_index += indices_ptr[ii] *input_stride[ii];
        }
        memcpy(output_data_ptr + output_index*ele_size,
               input_data_ptr + input_index*ele_size,
               ele_count * ele_size);
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(GatherND, LAYER_GATHERND);
}  // namespace TNN_NS
