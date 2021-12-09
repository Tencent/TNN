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

#include "tnn/core/blob_int8.h"
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/device/cpu/cpu_context.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

DECLARE_CPU_ACC(Concat, LAYER_CONCAT);

Status CpuConcatLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuConcatLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ConcatLayerParam *>(param_);
    if (!param) {
        LOGE("Error: ConcatLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: ConcatLayerParam is nil");
    }
    if (inputs.size() < 2) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "Concat layer's inputs size must >= 2");
    }
    auto input  = inputs[0];
    auto output = outputs[0];
    auto dims   = input->GetBlobDesc().dims;

    bool int8_per_tensor_flag = false;
    if (output->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        int8_per_tensor_flag = true;
        // if one blob is per channel quant, concat with the normal way
        for (auto &blob : inputs) {
            if (reinterpret_cast<BlobInt8 *>(blob)->GetIntResource()->scale_handle.GetDataCount() > 1) {
                int8_per_tensor_flag = false;
                break;
            }
        }
    }
    int axis = param->axis;
    if (axis < 0) {
        axis += (int)inputs[0]->GetBlobDesc().dims.size();
    }
    if (axis > dims.size() || axis < 0) {
        LOGE("Error: Concat layer param invalid\n");
        return Status(TNNERR_PARAM_ERR, "Concat layer param invalid");
    }

    int num_concats = 1;
    for (int i = 0; i < axis; i++) {
        num_concats *= dims[i];
    }

    int concate_size = 1;
    for (int i = axis + 1; i < dims.size(); i++) {
        concate_size *= dims[i];
    }

    auto datasize                 = DataTypeUtils::GetBytesSize(input->GetBlobDesc().data_type);
    int8_t *output_data           = static_cast<int8_t *>(output->GetHandle().base);
    int output_concat_axis        = output->GetBlobDesc().dims[axis];
    int output_concat_axis_offset = 0;

    if (!int8_per_tensor_flag) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            // use int8_t for all types
            int8_t *input_data          = static_cast<int8_t *>(inputs[i]->GetHandle().base);
            const int input_concat_axis = inputs[i]->GetBlobDesc().dims[axis];
            //support shape1[i] == 0 for empty blob in yolov5
            if (input_data) {
                for (int n = 0; n < num_concats; ++n) {
                    memcpy(output_data + (n * output_concat_axis + output_concat_axis_offset) * concate_size * datasize,
                           input_data + n * input_concat_axis * concate_size * datasize,
                           input_concat_axis * concate_size * datasize);
                }
            }
            output_concat_axis_offset += input_concat_axis;
        }
    } else {
        float *output_scale = reinterpret_cast<BlobInt8 *>(output)->GetIntResource()->scale_handle.force_to<float *>();
        int8_t *output_zero_point = reinterpret_cast<BlobInt8 *>(output)->GetIntResource()->zero_point_handle.force_to<int8_t *>();
        for (size_t i = 0; i < inputs.size(); ++i) {
            float *input_scale =
                reinterpret_cast<BlobInt8 *>(inputs[i])->GetIntResource()->scale_handle.force_to<float *>();
            int8_t *input_zero_point =
                reinterpret_cast<BlobInt8 *>(inputs[i])->GetIntResource()->zero_point_handle.force_to<int8_t *>();
            int8_t *input_data          = static_cast<int8_t *>(inputs[i]->GetHandle().base);
            const int input_concat_axis = inputs[i]->GetBlobDesc().dims[axis];
            for (int n = 0; n < num_concats; ++n) {
                int8_t *concat_dst = output_data + (n * output_concat_axis + output_concat_axis_offset) * concate_size;
                int8_t *concat_src = input_data + n * input_concat_axis * concate_size;
                // per tensor need dequant and requant
                for (int i = 0; i < input_concat_axis * concate_size; i++) {
                    float val = static_cast<float>(concat_src[i] - input_zero_point[0]);
                    concat_dst[i] = float2int8(val * input_scale[0] / output_scale[0] + output_zero_point[0]);
                }
            }
            output_concat_axis_offset += input_concat_axis;
        }
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(Concat, LAYER_CONCAT);

}  // namespace TNN_NS
