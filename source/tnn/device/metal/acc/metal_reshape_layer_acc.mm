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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
// ReshapeLayer has loaded constant input and set layer_param
DECLARE_METAL_ACC_WITH_EXTRA(Reshape, LAYER_RESHAPE, protected: virtual bool UseNaiveConstantBlobs(){return true;});

Status MetalReshapeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalReshapeLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto dims_input      = inputs[0]->GetBlobDesc().dims;
    auto dims_output     = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalReshapeParams metal_params;
        metal_params.input_width   = DimsFunctionUtils::GetDimProduct(dims_input, 3);
        metal_params.input_height  = DimsFunctionUtils::GetDim(dims_input, 2);
        metal_params.input_size    = metal_params.input_height * metal_params.input_width;
        metal_params.input_slice   = UP_DIV(dims_input[1], 4);
        metal_params.input_channel = dims_input[1];

        metal_params.output_width   = DimsFunctionUtils::GetDimProduct(dims_output, 3);
        metal_params.output_height  = DimsFunctionUtils::GetDim(dims_output, 2);
        metal_params.output_size    = metal_params.output_height * metal_params.output_width;
        metal_params.output_slice   = UP_DIV(dims_output[1], 4);
        metal_params.output_channel = dims_output[1];
        metal_params.batch          = dims_output[0];

        buffer_param_     = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(metal_params)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

std::string MetalReshapeLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<ReshapeLayerParam *>(param_);
    bool layout_nchw_ = (layer_param->reshape_type == 0);
    std::string kernel_name = "";
    if (layout_nchw_) {
        kernel_name = "reshape_common_nchw";
    } else {
        kernel_name = "reshape_common_nhwc";
    }
    return kernel_name;
}

Status MetalReshapeLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    auto output_width   = DimsFunctionUtils::GetDimProduct(dims_output, 3);
    auto output_height  = DimsFunctionUtils::GetDim(dims_output, 2);
    auto output_size    = output_height * output_width;
    auto output_slice   = UP_DIV(dims_output[1], 4);
    auto batch          = dims_output[0];
    size = MTLSizeMake(output_size, output_slice, batch);

    return TNN_OK;
}

Status MetalReshapeLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                     const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

Status MetalReshapeLayerAcc::SetKernelEncoderParam(
                                                   id<MTLComputeCommandEncoder> encoder,
                                                   const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

REGISTER_METAL_ACC(Reshape, LAYER_RESHAPE);
REGISTER_METAL_LAYOUT(LAYER_RESHAPE, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
