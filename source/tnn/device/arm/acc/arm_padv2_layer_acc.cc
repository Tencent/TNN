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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(PadV2, LAYER_PADV2);

static void ConstPadV2(float *input_data, float *output_data, DimsVector input_dims, DimsVector output_dims,
                       PadLayerParam *layer_param) {
    float value = layer_param->value;

    const int count = DimsVectorUtils::Count(output_dims);
    DimsVector output_dim_index(output_dims.size(), 0);
    for (int i = 0; i < count; i++) {
        auto input_index =
            DimsFunctionUtils::Pad(output_dim_index, input_dims, layer_param->pads, layer_param->type, nullptr);
        if (DimsFunctionUtils::IsInBox(input_index, input_dims)) {
            int input_offset = DimsOffsetUtils::ConvertIndexToOffset(input_dims, input_index);
            output_data[i]   = input_data[input_offset];
        } else {
            output_data[i] = value;
        }

        output_dim_index = DimsFunctionUtils::IncreaseIndex(output_dim_index, output_dims);
    }
}

static void ReflectPadV2(float *input_data, float *output_data, DimsVector input_dims, DimsVector output_dims,
                         PadLayerParam *layer_param) {
    const int count = DimsVectorUtils::Count(output_dims);
    DimsVector output_dim_index(output_dims.size(), 0);
    for (int i = 0; i < count; i++) {
        auto input_index =
            DimsFunctionUtils::Pad(output_dim_index, input_dims, layer_param->pads, layer_param->type, nullptr);
        int input_offset = DimsOffsetUtils::ConvertIndexToOffset(input_dims, input_index);
        output_data[i]   = input_data[input_offset];

        output_dim_index = DimsFunctionUtils::IncreaseIndex(output_dim_index, output_dims);
    }
}

Status ArmPadV2LayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PadLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: layer param is nil");
    }
    auto input_blob  = inputs[0];
    auto output_blob = outputs[0];

    auto input_dims  = input_blob->GetBlobDesc().dims;
    auto output_dims = output_blob->GetBlobDesc().dims;

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT ||
        output_blob->GetBlobDesc().data_type == DATA_TYPE_INT32 ||
        output_blob->GetBlobDesc().data_type == DATA_TYPE_UINT32) {
        auto input_data = reinterpret_cast<float*>(GetBlobHandlePtr(input_blob->GetHandle()));
        auto output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));

        if (layer_param->type == 0) {
            // mode: const
            ConstPadV2(input_data, output_data, input_dims, output_dims, layer_param);
        } else if (layer_param->type == 1) {
            // mode: reflect
            ReflectPadV2(input_data, output_data, input_dims, output_dims, layer_param);
        } else {
            LOGE("Error: CpuPadV2LayerAcc layer param is not supported: type:%d\n", layer_param->type);
            return Status(TNNERR_PARAM_ERR, "Error: CpuPadV2LayerAcc layer param is not supported");
        }
    } else if (output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        LOGE("Error: CpuPadV2LayerAcc layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuPadV2LayerAcc layer acc dont support datatype");
    } else {
        LOGE("Error: CpuPadV2LayerAcc layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuPadV2LayerAcc layer acc dont support datatype");
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(PadV2, LAYER_PADV2);
REGISTER_ARM_LAYOUT(LAYER_PADV2, DATA_FORMAT_NCHW);

}  // namespace TNN_NS