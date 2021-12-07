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

#include <cmath>
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC_WITH_FUNC(Tile, LAYER_REPEAT,
                          virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
                                                                 const std::vector<Blob *> &outputs););

Status CpuTileLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuTileLayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
                                                         const std::vector<Blob *> &outputs) {
    auto *layer_param = dynamic_cast<TileLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (inputs.size() >= 2) {
        if (inputs[1]->GetBlobDesc().data_type != DATA_TYPE_INT32) {
            return Status(TNNERR_PARAM_ERR, "TileLayer input(reps) has invalid data type");
        }
        auto dim_count = DimsVectorUtils::Count(inputs[1]->GetBlobDesc().dims);
        auto dim_data = (int *)((char *)inputs[1]->GetHandle().base + inputs[1]->GetHandle().bytes_offset);
        DimsVector dims;
        for (int i=0; i<dim_count; i++) {
            dims.push_back(dim_data[i]);
        }
        layer_param->reps = dims;
    }
    
    
    auto input_dims = inputs[0]->GetBlobDesc().dims;
    auto reps = layer_param->reps;
    
    auto output_dims = DimsFunctionUtils::Tile(input_dims, reps);
    
    outputs[0]->GetBlobDesc().dims = output_dims;
    
    return TNN_OK;
}

template <typename T>
Status Exec(Blob *input_blob, Blob *output_blob, const DimsVector &input_dims, const DimsVector &output_dims) {
    T *input_data  = static_cast<T *>(input_blob->GetHandle().base);
    T *output_data = static_cast<T *>(output_blob->GetHandle().base);
    int count      = DimsVectorUtils::Count(output_dims);
    DimsVector output_index(output_dims.size(), 0);
    for (int i = 0; i < count; i++) {
        auto input_index = DimsFunctionUtils::ModIndex(output_index, input_dims);
        int input_offset = DimsOffsetUtils::ConvertIndexToOffset(input_dims, input_index);
        output_data[i]   = input_data[input_offset];
        output_index     = DimsFunctionUtils::IncreaseIndex(output_index, output_dims);
    }
    return TNN_OK;
}

Status CpuTileLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<TileLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    Blob *input_blob       = inputs[0];
    Blob *output_blob      = outputs[0];
    auto input_dims        = input_blob->GetBlobDesc().dims;
    const auto output_dims = output_blob->GetBlobDesc().dims;
    while (input_dims.size() < output_dims.size()) {
        input_dims.insert(input_dims.begin(), 1);
    }

    auto data_type = output_blob->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(input_blob, output_blob, input_dims, output_dims);
    } else if (data_type == DATA_TYPE_HALF) {
        return Exec<fp16_t>(input_blob, output_blob, input_dims, output_dims);
    } else if (data_type == DATA_TYPE_INT32) {
        return Exec<int32_t>(input_blob, output_blob, input_dims, output_dims);
    } else {
        return Status(Status(TNNERR_MODEL_ERR, "CpuTileLayerAcc input has invalid data type"));
    }
}

REGISTER_CPU_ACC(Tile, LAYER_REPEAT);

}  // namespace TNN_NS
