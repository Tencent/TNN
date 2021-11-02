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

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(SplitV, LAYER_SPLITV);

Status CpuSplitVLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuSplitVLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<SplitVLayerParam *>(param_);
    if (!layer_param || layer_param->slices.size() != outputs.size()) {
        return Status(TNNERR_PARAM_ERR, "CpuSplitVLayerAcc has invalid param, slices size != output blobs size");
    }

    const int axis  = layer_param->axis;
    auto input_blob = inputs[0];
    auto input_dims = input_blob->GetBlobDesc().dims;
    const int bath  = DimsVectorUtils::Count(input_dims, 0, axis);
    int slice_size  = DimsVectorUtils::Count(input_dims, axis + 1);
    if (slice_size == 0) {
        // support split empty blob such as blob with shape[12, 2, 128], axis = 0
        return TNN_OK;
    }
    const int slice_input = input_dims[axis];
    auto input_data       = static_cast<float *>(input_blob->GetHandle().base);

    if (input_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        for (size_t b = 0; b < bath; b++) {
            int slice_input_offset = 0;
            for (size_t i = 0; i < outputs.size(); i++) {
                auto output_blob = outputs[i];
                auto output_data = static_cast<float *>(output_blob->GetHandle().base);
                const int slice  = output_blob->GetBlobDesc().dims[axis];

                auto input_data_ptr  = input_data + b * slice_input * slice_size + slice_input_offset * slice_size;
                auto output_data_ptr = output_data + b * slice * slice_size;

                memcpy(output_data_ptr, input_data_ptr, slice * slice_size * sizeof(float));
                slice_input_offset += slice;
            }
        }
    } else if (input_blob->GetBlobDesc().data_type == DATA_TYPE_INT32) {
        for (size_t b = 0; b < bath; b++) {
            int slice_input_offset = 0;
            for (size_t i = 0; i < outputs.size(); i++) {
                auto output_blob = outputs[i];
                auto output_data = static_cast<int *>(output_blob->GetHandle().base);
                const int slice  = output_blob->GetBlobDesc().dims[axis];

                auto input_data_ptr  = input_data + b * slice_input * slice_size + slice_input_offset * slice_size;
                auto output_data_ptr = output_data + b * slice * slice_size;

                memcpy(output_data_ptr, input_data_ptr, slice * slice_size * sizeof(int));
                slice_input_offset += slice;
            }
        }
    } else if (input_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        LOGE("Error: CpuSplitVLayerAcc layer acc don't support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuSplitVLayerAcc layer acc dont support datatype");
    } else {
        LOGE("Error: CpuSplitVLayerAcc layer acc dont support datatype: %d\n", input_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuSplitVLayerAcc layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(SplitV, LAYER_SPLITV);

}  // namespace TNN_NS
