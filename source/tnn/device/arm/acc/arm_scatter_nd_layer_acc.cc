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
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(ScatterND, LAYER_SCATTER_ND);

Status ArmScatterNDLayerAcc::DoForward(const std::vector<Blob*>& inputs, const std::vector<Blob*>& outputs) {
    auto resource = dynamic_cast<ScatterNDLayerResource*>(resource_);
    if (!resource && inputs.size() < 3) {
        LOGE("ArmScatterNDLayerAcc has not layer resource\n");
        return Status(TNNERR_PARAM_ERR, "ArmScatterNDLayerAcc has not layer resource");
    }

    DimsVector indices_dims;
    int* indice_offset     = nullptr;
    Blob* update_data_blob = nullptr;
    if (inputs.size() >= 3) {
        if (inputs[1]->GetBlobDesc().data_type != DATA_TYPE_INT32) {
            LOGE("ArmScatterNDLayerAcc indice input has invalid data type\n");
            return Status(TNNERR_PARAM_ERR, "ArmScatterNDLayerAcc indice input has invalid data type");
        }
        indice_offset    = reinterpret_cast<int*>(GetBlobHandlePtr(inputs[1]->GetHandle()));
        indices_dims     = inputs[1]->GetBlobDesc().dims;
        update_data_blob = inputs[2];
    } else {
        indice_offset    = resource->indices.force_to<int*>();
        indices_dims     = resource->indices.GetBufferDims();
        update_data_blob = inputs[1];
    }

    Blob* output_blob = outputs[0];
    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        Blob* input_data_blob = inputs[0];
        float* input_data     = reinterpret_cast<float*>(GetBlobHandlePtr(input_data_blob->GetHandle()));
        float* update_data    = reinterpret_cast<float*>(GetBlobHandlePtr(update_data_blob->GetHandle()));
        float* output_data    = reinterpret_cast<float*>(GetBlobHandlePtr(output_blob->GetHandle()));
        auto input_dims       = input_data_blob->GetBlobDesc().dims;
        auto update_dims      = update_data_blob->GetBlobDesc().dims;
        if (indices_dims.empty()) {
            LOGE("Error: indices dims has rank 0");
            return Status(TNNERR_PARAM_ERR, "Error: indices dims has rank 0");
        }
        auto indice_rank           = indices_dims.size();
        auto last_indice_dimension = indices_dims[indice_rank - 1];
        if (last_indice_dimension > input_dims.size()) {
            LOGE("Error: last dimension of indices larger than input blob dims size ");
            return Status(TNNERR_PARAM_ERR, "Error: last dimension of indices larger than input blob dims size ");
        }

        auto update_rank = update_dims.size();
        auto input_rank  = input_dims.size();
        if (update_rank < indice_rank - 1) {
            LOGE("Error: update_rank < indice_rank -1 ");
            return Status(TNNERR_PARAM_ERR, "Error: update_rank < indice_rank -1 ");
        }

        for (int i = 0; i < indice_rank - 1; ++i) {
            if (indices_dims[i] != update_dims[i]) {
                LOGE("Error: indices_dims and update dims not equal before index indice_rank -1");
                return Status(TNNERR_PARAM_ERR,
                              "Error: indices_dims and update dims not equal before index indice_rank -1");
            }
        }

        if (DimsVectorUtils::Count(update_dims, indice_rank - 1) !=
            DimsVectorUtils::Count(input_dims, last_indice_dimension)) {
            LOGE("Error: indices_dims and update dims not equal before index indice_rank -1");
            return Status(TNNERR_PARAM_ERR,
                          "Error: indices_dims and update dims not equal before index indice_rank -1");
        }

        // copy input to output
        memcpy(output_data, input_data, DimsVectorUtils::Count(input_dims) * sizeof(float));

        std::vector<int> element_counts(last_indice_dimension, 0);

        for (int i = 0; i < last_indice_dimension; ++i) {
            element_counts[i] = DimsVectorUtils::Count(input_dims, i + 1);
        }

        int element_to_copy = DimsVectorUtils::Count(input_dims, last_indice_dimension);
        int offset_count    = DimsVectorUtils::Count(indices_dims, 0, indice_rank - 1);

        for (int i = 0; i < offset_count; ++i) {
            int offset = 0;
            for (int j = 0; j < last_indice_dimension; ++j) {
                auto indice = *(indice_offset + i * last_indice_dimension + j);
                offset += indice * element_counts[j];
            }
            memcpy(output_data + offset, update_data + i * element_to_copy, element_to_copy * sizeof(float));
        }

        return TNN_OK;

    } else {
        LOGE("Error: ArmScatterNDLayerAcc layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: ArmScatterNDLayerAcc layer acc dont support datatype");
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(ScatterND, LAYER_SCATTER_ND);
REGISTER_ARM_LAYOUT(LAYER_SCATTER_ND, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
