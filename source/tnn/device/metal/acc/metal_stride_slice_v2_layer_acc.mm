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

#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/acc/metal_layer_acc.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/half_utils_inner.h"

namespace TNN_NS {

DECLARE_METAL_ACC(StrideSliceV2, LAYER_STRIDED_SLICE);

Status MetalStrideSliceV2LayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalStrideSliceV2LayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                     const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<StrideSliceV2LayerParam *>(param_);
    if (!layer_param) {
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParamV2 is nil");
    }

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    auto dim_size    = dims_output.size();
    if (dim_size > 4 || dim_size != dims_input.size()) {
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParamV2 not support!");
    }

    auto begins = layer_param->begins;
    auto ends = layer_param->ends;
    auto strides = layer_param->strides;
    auto axes = layer_param->axes;

    Status status = TNN_OK;
    DimsFunctionUtils::StrideSlice(dims_input, begins, ends, strides, axes, &status);
    RETURN_ON_NEQ(status, TNN_OK);
    // buffer_param_
    {
        MetalStrideSliceParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);

        std::vector<int> rectified_begins(dim_size, 0);
        std::vector<int> rectified_strides(dim_size, 0);

        for(int i=0, axes_idx=0; i<dims_output.size(); ++i) {
            if (axes_idx >= axes.size() || i != axes[axes_idx]) {
                rectified_begins[i]  = 0;
                rectified_strides[i] = 1;
            } else {
                rectified_begins[i]  = begins[axes_idx];
                rectified_strides[i] = strides[axes_idx];
                axes_idx += 1;
            }
        }
        // pad to size 4
        for(int i=dims_output.size(); i<4; ++i) {
            rectified_begins.push_back(0);
            rectified_strides.push_back(1);
        }

        metal_params.begin_n = rectified_begins[0];
        metal_params.begin_c = rectified_begins[1];
        metal_params.begin_h = rectified_begins[2];
        metal_params.begin_w = rectified_begins[3];

        metal_params.stride_n = rectified_strides[0];
        metal_params.stride_c = rectified_strides[1];
        metal_params.stride_h = rectified_strides[2];
        metal_params.stride_w = rectified_strides[3];

        buffer_param_         = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalStrideSliceParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

std::string MetalStrideSliceV2LayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "stride_slice_common";
}

Status MetalStrideSliceV2LayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

Status MetalStrideSliceV2LayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

Status MetalStrideSliceV2LayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
       auto dims_output = outputs[0]->GetBlobDesc().dims;
       size = GetDefaultThreadSize(dims_output, false);
       return TNN_OK;
}

REGISTER_METAL_ACC(StrideSliceV2, LAYER_STRIDED_SLICE_V2);
REGISTER_METAL_LAYOUT(LAYER_STRIDED_SLICE_V2, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
