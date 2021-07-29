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
#include "tnn/utils/half_utils_inner.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status IsMetalStrideSliceLayerAccSupported(LayerParam *param, LayerResource *resource,
                                           const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    auto layer_param = dynamic_cast<StrideSliceLayerParam *>(param);
    if (!layer_param) {
        LOGE("Error: StrideSliceLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParam is nil");
    }

    //    if (layer_param->begins[2] %4 !=0) {
    //        LOGE("Error: StrideSlice's begins channel must be 4x\n");
    //        return Status(TNNERR_NET_ERR, "StrideSlice's begins channel must be 4x\n");
    //    }
    return TNN_OK;
}

DECLARE_METAL_ACC(StrideSlice, LAYER_STRIDED_SLICE);

Status MetalStrideSliceLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = IsMetalStrideSliceLayerAccSupported(param_, resource_, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }

    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalStrideSliceLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                     const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<StrideSliceLayerParam *>(param_);
    if (!layer_param) {
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParam is nil");
    }

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalStrideSliceParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);

        auto begins = layer_param->begins;
        std::reverse(begins.begin(), begins.end());
        metal_params.begin_n = begins[0];
        metal_params.begin_c = begins[1];
        metal_params.begin_h = begins[2];
        metal_params.begin_w = begins[3];

        auto strides = layer_param->strides;
        std::reverse(strides.begin(), strides.end());
        metal_params.stride_n = strides[0];
        metal_params.stride_c = strides[1];
        metal_params.stride_h = strides[2];
        metal_params.stride_w = strides[3];
        buffer_param_         = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalStrideSliceParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

std::string MetalStrideSliceLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "stride_slice_common";
}

Status MetalStrideSliceLayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

Status MetalStrideSliceLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = IsMetalStrideSliceLayerAccSupported(param_, resource_, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    
    return MetalLayerAcc::Forward(inputs, outputs);
}

Status MetalStrideSliceLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
       auto dims_output = outputs[0]->GetBlobDesc().dims;
       size = GetDefaultThreadSize(dims_output, false);
       return TNN_OK;
}

REGISTER_METAL_ACC(StrideSlice, LAYER_STRIDED_SLICE);
REGISTER_METAL_LAYOUT(LAYER_STRIDED_SLICE, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
