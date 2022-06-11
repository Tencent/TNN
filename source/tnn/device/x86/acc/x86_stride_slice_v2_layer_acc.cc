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
#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/acc/x86_stride_slice_v2_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/dims_offset_utils.h"
#include "tnn/device/x86/acc/compute/x86_compute.h"

namespace TNN_NS {

X86StrideSliceV2LayerAcc::~X86StrideSliceV2LayerAcc() {}

Status X86StrideSliceV2LayerAcc::InferRuntimeOutputShape(const std::vector<Blob *> &inputs,
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
    
    Status status = TNN_OK;
    auto output_dims = DimsFunctionUtils::StrideSlice(input_dims, begins, ends, strides, axes, &status);
    RETURN_ON_NEQ(status, TNN_OK);
    
    outputs[0]->GetBlobDesc().dims = output_dims;
    
    return TNN_OK;
}


Status X86StrideSliceV2LayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<StrideSliceV2LayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: StrideSliceLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];

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
    RETURN_ON_NEQ(status, TNN_OK);

    DimsVector begins_compute;
    DimsVector strides_compute;
    begins_compute.reserve(output_dims.size());
    strides_compute.reserve(output_dims.size());
    for (int i = 0, axes_index = 0; i < output_dims.size(); i++) {
        if (axes_index < axes.size() && i == axes[axes_index]) {
            begins_compute.push_back(begins[axes_index]);
            strides_compute.push_back(strides[axes_index]);
            ++axes_index;
        }  else {
            begins_compute.push_back(0);
            strides_compute.push_back(1);
        }
    }

    DimsVector input_strides;
    DimsVector output_strides;
    input_strides.reserve(output_dims.size());
    output_strides.reserve(output_dims.size());

    for (int i = 0; i < output_dims.size() - 1; i++) {
        input_strides.push_back(DimsVectorUtils::Count(input_dims, i + 1));
        output_strides.push_back(DimsVectorUtils::Count(output_dims, i + 1));
    }

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = handle_ptr<float *>(input_blob->GetHandle());
        float *output_data = handle_ptr<float *>(output_blob->GetHandle());

        X86StrideSliceImpl(begins_compute, strides_compute, output_dims, input_strides, output_strides, input_data, output_data);
    } else {
        return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR int8/bfp16 StrideSliceV2");
    }

    return TNN_OK;
}

REGISTER_X86_ACC(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

}  // namespace TNN_NS
