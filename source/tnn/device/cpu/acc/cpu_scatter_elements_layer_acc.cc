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
// COElementsITIONS OF ANY KIElements, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <algorithm>
#include <cmath>

#include "tnn/device/cpu/acc/cpu_layer_acc.h"

#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(ScatterElements, LAYER_SCATTER_ELEMENTS);

Status CpuScatterElementsLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuScatterElementsLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ScatterElementsLayerParam *>(param_);

    DimsVector data_dims = inputs[0]->GetBlobDesc().dims;
    DimsVector indices_dims = inputs[1]->GetBlobDesc().dims;
    DimsVector update_dims = inputs[2]->GetBlobDesc().dims;
    float* data_ptr = (float*)inputs[0]->GetHandle().base;

    int* indices_offset = (int *)inputs[1]->GetHandle().base;
    float* update_data = (float*)inputs[2]->GetHandle().base;
    int total_input_bytes = DimsVectorUtils::Count(data_dims) * sizeof(float);

    auto num_indices = DimsVectorUtils::Count(indices_dims);

    std::vector<int> indices_data;
    indices_data.reserve(num_indices);

    auto axis_dim_limit = data_dims[param->axis];
    for (int i = 0; i < num_indices; ++i) {
        int idx = indices_offset[i];
        indices_data.push_back(idx < 0 ? idx + axis_dim_limit : idx);
    }
    auto input_elements = DimsVectorUtils::Count(data_dims);
    float* output_data = reinterpret_cast<float*>(outputs[0]->GetHandle().base);

    memcpy(output_data, data_ptr, total_input_bytes);

    auto num_dims = data_dims.size();

    std::vector<int> dims_counters(num_dims);
    std::vector<int> dim_block_size(num_dims);

    dim_block_size.back() = 1;
    if (num_dims > 1) {
        for (auto i = int(num_dims - 2);  i >= 0; --i) {
            dim_block_size[i] = data_dims[i + 1] * dim_block_size[i + 1];
        }
    }
    for (int index = 0; index < num_indices;) {
        int axis_idx = indices_data[index];
        size_t dst_offset = 0;
        for (size_t i = 0; i < num_dims; ++i) {
            if (i == size_t(param->axis)) {
                dst_offset += axis_idx * dim_block_size[i];
            } else {
                dst_offset += dims_counters[i] * dim_block_size[i];
            }
        }
        if (param->op == 0) {
            output_data[dst_offset] = update_data[index];
        } else {
            output_data[dst_offset] += update_data[index];
        }

        if (++index == num_indices) {
            break;
        }

        for (int i = num_dims - 1; i >= 0; --i) {
            auto v = ++dims_counters[i];
            if (v < update_dims[i]) {
                break;
            }
            dims_counters[i] = 0;
        }
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(ScatterElements, LAYER_SCATTER_ELEMENTS);

}  // namespace TNN_NS
