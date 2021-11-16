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

DECLARE_CPU_ACC_WITH_FUNC(StrideSliceV2, LAYER_STRIDED_SLICE_V2,
                          virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
                                                                 const std::vector<Blob *> &outputs););

Status CpuStrideSliceV2LayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuStrideSliceV2LayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
                                                         const std::vector<Blob *> &outputs) {
    auto *layer_param = dynamic_cast<StrideSliceV2LayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    
    if (inputs.size() >= 2) {
        if (inputs[1]->GetBlobDesc().data_type != DATA_TYPE_INT32) {
            return Status(TNNERR_PARAM_ERR, "stride slice input(begins) has invalid data type");
        }
        auto dim_count = DimsVectorUtils::Count(inputs[1]->GetBlobDesc().dims);
        auto dim_data = (int *)((char *)inputs[1]->GetHandle().base + inputs[1]->GetHandle().bytes_offset);
        DimsVector dims;
        for (int i=0; i<dim_count; i++) {
            dims.push_back(dim_data[i]);
        }
        layer_param->begins = dims;
    }
    
    if (inputs.size() >= 3) {
        if (inputs[2]->GetBlobDesc().data_type != DATA_TYPE_INT32) {
            return Status(TNNERR_PARAM_ERR, "stride slice input(ends) has invalid data type");
        }
        auto input_dims = inputs[2]->GetBlobDesc().dims;
        
        auto dim_count = DimsVectorUtils::Count(inputs[2]->GetBlobDesc().dims);
        auto dim_data = (int *)((char *)inputs[2]->GetHandle().base + inputs[2]->GetHandle().bytes_offset);
        DimsVector dims;
        for (int i=0; i<dim_count; i++) {
            dims.push_back(dim_data[i]);
        }
        layer_param->ends = dims;
    }
    
    auto input_dims = inputs[0]->GetBlobDesc().dims;
    
    auto begins = layer_param->begins;
    auto ends = layer_param->ends;
    auto axes = layer_param->axes;
    auto strides = layer_param->strides;
    
    //前闭后开区间
    Status status = TNN_OK;
    auto output_dims = DimsFunctionUtils::StrideSlice(input_dims, begins, ends, strides, axes, &status);
    //support empty blob for yolov5 Slice_507, only in device cpu
    if (status != TNN_OK && !(output_dims.size() == input_dims.size() &&  runtime_model_ == RUNTIME_MODE_CONST_FOLD)) {
        return status;
    }
    
    outputs[0]->GetBlobDesc().dims = output_dims;
    
    return TNN_OK;
}

template <typename T>
void StrideSlice(Blob *input_blob, Blob *output_blob, const DimsVector &begins, const DimsVector &axes,
                 const DimsVector &strides) {
    DimsVector input_dims  = input_blob->GetBlobDesc().dims;
    DimsVector output_dims = output_blob->GetBlobDesc().dims;
    const int output_count = DimsVectorUtils::Count(output_dims);

    T *input_data  = static_cast<T *>(input_blob->GetHandle().base);
    T *output_data = static_cast<T *>(output_blob->GetHandle().base);
    for (int offset = 0; offset < output_count; ++offset) {
        DimsVector output_index = DimsOffsetUtils::ConvertOffsetToIndex(output_dims, offset);
        DimsVector input_index;
        int axes_index = 0;
        for (int i = 0; i < output_index.size(); ++i) {
            if (axes_index < axes.size() && i == axes[axes_index]) {
                input_index.push_back(begins[axes_index] + output_index[i] * strides[axes_index]);
                ++axes_index;
            } else {
                input_index.push_back(output_index[i]);
            }
        }
        int in_offset       = DimsOffsetUtils::ConvertIndexToOffset(input_dims, input_index);
        output_data[offset] = input_data[in_offset];
    }
}

Status CpuStrideSliceV2LayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<StrideSliceV2LayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: StrideSliceLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];

    const int input_dims_size = input_blob->GetBlobDesc().dims.size();
    for (auto &axis : layer_param->axes) {
        if (axis < 0) {
            axis += input_dims_size;
        }
    }

    auto begins = layer_param->begins;
    auto ends = layer_param->ends;
    auto strides = layer_param->strides;
    auto axes = layer_param->axes;
    
    DimsVector input_dims = input_blob->GetBlobDesc().dims;
    DimsVector output_dims = output_blob->GetBlobDesc().dims;
    int output_count = DimsVectorUtils::Count(output_dims);
    
    //rectify begins and ends here for value < 0 or = INT_MAX
    Status status = TNN_OK;
    DimsFunctionUtils::StrideSlice(input_dims, begins, ends, strides, axes, &status);
    //support empty blob for yolov5 Slice_507, only in device cpu
    if (status != TNN_OK && !(output_dims.size() == input_dims.size() &&  runtime_model_ == RUNTIME_MODE_CONST_FOLD)) {
        return status;
    }

    if (output_blob->GetBlobDesc().data_type != DATA_TYPE_INT8) {
        if (output_blob->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            StrideSlice<fp16_t>(input_blob, output_blob, begins, axes, strides);
        } else {
            StrideSlice<float>(input_blob, output_blob, begins, axes, strides);
        }
    } else {
        ASSERT(0);
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

}  // namespace TNN_NS
