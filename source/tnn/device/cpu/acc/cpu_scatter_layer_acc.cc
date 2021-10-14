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

#include <algorithm>
#include <cmath>

#include "tnn/device/cpu/acc/cpu_layer_acc.h"

#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(Scatter, LAYER_SCATTER);

Status CpuScatterLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuScatterLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Blob *input_data_blob = inputs[0];
    auto input_data_dims  = input_data_blob->GetBlobDesc().dims;

    // check param
    auto param = dynamic_cast<ScatterLayerParam *>(param_);
    int rank   = input_data_dims.size();
    if (param->axis < -rank || param->axis >= rank) {
        LOGE("CpuScatterLayerAcc param->axis is not between [-rank,rank-1]\n");
        return Status(TNNERR_PARAM_ERR, "CpuScatterLayerAcc param->axis is not between [-rank,rank-1]");
    }
    int axis = param->axis < 0 ? param->axis + rank : param->axis;

    // check resource
    auto resource = dynamic_cast<ScatterLayerResource *>(resource_);
    if (!resource && inputs.size() < 3) {
        LOGE("CpuScatterLayerAcc has not layer resource\n");
        return Status(TNNERR_PARAM_ERR, "CpuScatterLayerAcc has not layer resource");
    }

    // input order: data, indics, updates
    DimsVector indices_dims;
    int *indice_offset     = nullptr;
    Blob *update_data_blob = nullptr;
    if (inputs.size() >= 3) {
        if (inputs[1]->GetBlobDesc().data_type != DATA_TYPE_INT32) {
            LOGE("CpuScatterLayerAcc indice input has invalid data type\n");
            return Status(TNNERR_PARAM_ERR, "CpuScatterLayerAcc indice input has invalid data type");
        }
        indice_offset    = (int *)((char *)inputs[1]->GetHandle().base + inputs[1]->GetHandle().bytes_offset);
        indices_dims     = inputs[1]->GetBlobDesc().dims;
        update_data_blob = inputs[2];
    } else {
        indice_offset    = resource->indices.force_to<int *>();
        indices_dims     = resource->indices.GetBufferDims();
        update_data_blob = inputs[1];
    }
    auto update_dims = update_data_blob->GetBlobDesc().dims;

    // check indices dim
    if (indices_dims.empty()) {
        LOGE("Error: indices dims has rank 0");
        return Status(TNNERR_PARAM_ERR, "Error: indices dims has rank 0");
    }

    // check data and update DataType
    if (input_data_blob->GetBlobDesc().data_type != update_data_blob->GetBlobDesc().data_type) {
        LOGE("CpuScatterLayerAcc the DataType of data and updates is not same\n");
        return Status(TNNERR_PARAM_ERR, "CpuScatterLayerAcc the DataType of data and updates is not same");
    }

    // check indics and updates dim
    if (!DimsVectorUtils::Equal(indices_dims, update_dims)) {
        LOGE("CpuScatterLayerAcc the dims of indics and updates is not same\n");
        return Status(TNNERR_PARAM_ERR, "CpuScatterLayerAcc the dims of indics and updates is not same");
    }

    // check data dims
    for (int i = 0; i < input_data_dims.size(); ++i) {
        if (i != axis && input_data_dims[i] < indices_dims[i]) {
            LOGE("CpuScatterLayerAcc the dims of indics is invalid (bigger than input_data dim)\n");
            return Status(TNNERR_PARAM_ERR,
                          "CpuScatterLayerAcc the dims of indics is invalid (bigger than input_data dim)");
        }
    }

    // check indics value
    int axis_dim_limit    = input_data_dims[axis];
    int indices_dim_count = DimsVectorUtils::Count(indices_dims);
    for (int i = 0; i < indices_dim_count; ++i) {
        if (indice_offset[i] < -axis_dim_limit || indice_offset[i] >= axis_dim_limit) {
            LOGE("CpuScatterLayerAcc the value of indics is invalid (bigger than input_data dim)\n");
            return Status(TNNERR_PARAM_ERR,
                          "CpuScatterLayerAcc the value of indics is invalid (bigger than input_data dim)");
        }

        indice_offset[i] = indice_offset[i] < 0 ? indice_offset[i] + axis_dim_limit : indice_offset[i];
    }

    Blob *output_blob = outputs[0];
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = reinterpret_cast<float *>(input_data_blob->GetHandle().base);
        float *update_data = reinterpret_cast<float *>(update_data_blob->GetHandle().base);
        float *output_data = reinterpret_cast<float *>(output_blob->GetHandle().base);

        // copy input to output
        memcpy(output_data, input_data, DimsVectorUtils::Count(input_data_dims) * sizeof(float));

        int dim_size = input_data_dims.size();
        std::vector<int> dim_block_size(dim_size);
        dim_block_size.back() = 1;
        if (dim_size > 1) {
            // start at num_dims - 2 because we already pre-populated the last element above
            for (int i = dim_size - 2; i >= 0; --i) {
                dim_block_size[i] = input_data_dims[i + 1] * dim_block_size[i + 1];
            }
        }

        std::vector<int> dim_counter(dim_size, 0);

        for (int idx = 0; idx < indices_dim_count; ++idx) {
            int axis_idx      = indice_offset[idx];
            int output_offset = 0;

            for (int d = 0; d < dim_size; ++d) {
                if (d == axis) {
                    output_offset += dim_block_size[d] * axis_idx;
                } else {
                    output_offset += dim_block_size[d] * dim_counter[d];
                }
            }

            output_data[output_offset] = update_data[idx];

            // increase dim_counter
            for (auto i = dim_size - 1; i >= 0; --i) {
                auto v = ++dim_counter[i];
                assert(v <= update_dims[i]);
                if (v < update_dims[i]) {
                    break;
                }
                assert(i > 0);
                dim_counter[i] = 0;
            }
        }

        return TNN_OK;

    } else {
        LOGE("Error: CpuScatterLayerAcc layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuScatterLayerAcc layer acc dont support datatype");
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Scatter, LAYER_SCATTER);

}  // namespace TNN_NS
