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

namespace TNN_NS {

DECLARE_METAL_ACC(Normalize, LAYER_NORMALIZE);

Status MetalNormalizeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalNormalizeLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto layer_param     = dynamic_cast<NormalizeLayerParam *>(param_);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalNormalizeParams metal_params;
        metal_params.output_width  = dims_output[3];
        metal_params.output_height = dims_output[2];
        metal_params.output_size   = metal_params.output_height * metal_params.output_width;
        metal_params.output_slice  = UP_DIV(dims_output[1], 4);

        metal_params.batch = dims_output[0];

        metal_params.epsilon = layer_param->epsilon;

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalNormalizeParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

std::string MetalNormalizeLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "";
}

Status MetalNormalizeLayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

Status MetalNormalizeLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    return MetalLayerAcc::ComputeThreadSize(inputs, outputs, size);
}

Status MetalNormalizeLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param  = dynamic_cast<NormalizeLayerParam *>(param_);
    auto context_impl = context_->getMetalContextImpl();
    auto encoder      = [context_impl encoder];
    encoder.label = GetKernelLabel();

    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_output    = output->GetBlobDesc().dims;
    auto output_width   = dims_output[3];
    auto output_height  = dims_output[2];
    auto output_channel = dims_output[1];
    auto output_slice   = UP_DIV(dims_output[1], 4);
    auto batch          = dims_output[0];
    auto mode           = dims_output[1] % 4;

    MetalBandwidth bandwidth;
    Status status        = TNN_OK;
    DataType data_type   = output->GetBlobDesc().data_type;
    string data_type_str = DataTypeUtils::GetDataTypeString(data_type);

    do {
        if (layer_param->axis == 1) {
            if (output_slice == 1) {
                status = [context_impl load:[NSString stringWithFormat:@"normaliz_%d_axis_1_slice_1_channel_%d",
                                                                       layer_param->p, output_channel]
                                    encoder:encoder
                                  bandwidth:bandwidth];
            } else {
                status = [context_impl
                         load:[NSString stringWithFormat:@"normaliz_%d_axis_1_common_channel_%d", layer_param->p, mode]
                      encoder:encoder
                    bandwidth:bandwidth];
            }
        } else {
            LOGE("MetalNormalizeLayerAcc do not support axis!=1\n");
            status = Status(TNNERR_LAYER_ERR, "MetalNormalizeLayerAcc do not support axis!=1");
        }
        BREAK_IF(status != TNN_OK);

        MTLSize threads = {(NSUInteger)output_width * output_height, 1, (NSUInteger)batch};

        status = SetKernelEncoderParam(encoder, inputs, outputs);
        BREAK_IF(status != TNN_OK);

        status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
    } while (0);

    [encoder endEncoding];
    [context_impl commit];
    TNN_PRINT_ENCODER(context_, encoder, this);
    return status;
}

REGISTER_METAL_ACC(Normalize, LAYER_NORMALIZE);

} // namespace TNN_NS
