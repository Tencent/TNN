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
    
    if (dim_size != dims_input.size()) {
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParamV2 not support different dim_size!");
    }
    if (dim_size > 6) {
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParamV2 not support dim_size>6!");
    }
    auto begins  = layer_param->begins;
    auto ends    = layer_param->ends;
    auto strides = layer_param->strides;
    auto axes    = layer_param->axes;

    Status status = TNN_OK;
    DimsFunctionUtils::StrideSlice(dims_input, begins, ends, strides, axes, &status);
    RETURN_ON_NEQ(status, TNN_OK);

    if (dim_size <= 4) {
        // use strideslice for 4 dim
        // buffer_param_
        {
            MetalStrideSliceParams metal_params;
            SetDefaultMetalParams(metal_params, dims_input, dims_output);

            std::vector<int> rectified_begins(dim_size, 0);
            std::vector<int> rectified_strides(dim_size, 1);

            for (int i = 0, axes_idx = 0; i < dims_output.size(); ++i) {
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
            for (int i = dims_output.size(); i < 4; ++i) {
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

            buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                                length:sizeof(MetalStrideSliceParams)
                                               options:MTLResourceCPUCacheModeWriteCombined];
        }
    } else {
        const int max_dim = 6;
        // stride slice for high dimension
        MetalStrideSliceParamsV2 metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        metal_params.input_width  = DimsFunctionUtils::GetDimProduct(dims_input, 3);
        metal_params.output_width = DimsFunctionUtils::GetDimProduct(dims_output, 3);
        metal_params.input_size  = metal_params.input_width * metal_params.input_height;
        metal_params.output_size = metal_params.output_width * metal_params.output_height;
        for(int i=0; i<3; ++i)
            metal_params.input_shape3d_low[3-i-1] = DimsFunctionUtils::GetDim(dims_input, 3+i);
        for(int i=0; i<3; ++i)
            metal_params.shape3d_low[3-i-1] = DimsFunctionUtils::GetDim(dims_output, 3+i);
        
        std::vector<int> rectified_begins(max_dim, 0);
        std::vector<int> rectified_strides(max_dim, 1);
        for (int i = 0, axes_idx = 0; i < max_dim; ++i) {
            if (axes_idx >= axes.size() || i != axes[axes_idx]) {
                rectified_begins[i]  = 0;
                rectified_strides[i] = 1;
            } else {
                rectified_begins[i]  = begins[axes_idx];
                rectified_strides[i] = strides[axes_idx];
                axes_idx += 1;
            }
        }
        // pad to max_dim
        for (int i = dims_output.size(); i < max_dim; ++i) {
            rectified_begins.push_back(0);
            rectified_strides.push_back(1);
        }

        for(int i=0; i<max_dim; ++i) {
            if (i < max_dim/ 2) {
                metal_params.strides_low[i] = rectified_strides[max_dim-1-i];
                metal_params.begins_low[i]  = rectified_begins[max_dim-1-i];
            } else {
                metal_params.strides_high[i-max_dim/2] = rectified_strides[max_dim-1-i];
                metal_params.begins_high[i-max_dim/2]  = rectified_begins[max_dim-1-i];
            }
        }

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                                length:sizeof(MetalStrideSliceParamsV2)
                                               options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

std::string MetalStrideSliceV2LayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    const auto dims_size = outputs[0]->GetBlobDesc().dims.size();
    if (dims_size <= 4)
        return "stride_slice_common";
    return "stride_slice_common_dim6";
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
       auto dims_output  = outputs[0]->GetBlobDesc().dims;
       const auto batch  = DimsFunctionUtils::GetDim(dims_output,0);
       const auto slice  = UP_DIV(DimsFunctionUtils::GetDim(dims_output,1), 4);
       const auto height = DimsFunctionUtils::GetDim(dims_output,2);
       const auto isize  = DimsFunctionUtils::GetDimProduct(dims_output, 3);
       size = MTLSizeMake(isize, height, batch * slice);
       return TNN_OK;
}

REGISTER_METAL_ACC(StrideSliceV2, LAYER_STRIDED_SLICE_V2);
REGISTER_METAL_LAYOUT(LAYER_STRIDED_SLICE_V2, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
