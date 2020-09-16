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

#include "tnn/device/metal/acc/metal_prelu_layer_acc.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils.h"

namespace TNN_NS {
MetalPReluLayerAcc::~MetalPReluLayerAcc() {}

Status MetalPReluLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PReluLayerParam *>(param_);
    if (!layer_param) {
        return Status(TNNERR_MODEL_ERR, "Error: PReluLayerParam is nil");
    }
    auto layer_res = dynamic_cast<PReluLayerResource *>(resource_);
    if (!layer_res) {
        return Status(TNNERR_MODEL_ERR, "Error: PReluLayerResource is nil");
    }

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        auto metal_params          = GetDefaultMetalParams(dims_input, dims_output);
        metal_params.share_channel = layer_param->channel_shared;
        buffer_param_              = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(metal_params)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    Status status = TNN_OK;
    if (!buffer_slope_) {
        buffer_slope_ = AllocateMetalBufferFormRawBuffer1D(layer_res->slope_handle, dims_output[1], status);
    }
    return status;
}

Status MetalPReluLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto context_impl = context_->getMetalContextImpl();
    auto encoder      = [context_impl encoder];
    encoder.label = GetKernelLabel();

    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_output  = output->GetBlobDesc().dims;
    auto batch        = dims_output[0];
    auto output_width = dims_output[3], output_height = dims_output[2],
         output_slice = UP_DIV(dims_output[1], 4) * dims_output[0];

    Status status = TNN_OK;
    MetalBandwidth bandwidth;
    do {
        status = [context_impl load:@"prelu" encoder:encoder bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);

        MTLSize threads = {(NSUInteger)output_width * output_height, (NSUInteger)output_slice, (NSUInteger)batch};

        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                    offset:(NSUInteger)input->GetHandle().bytes_offset
                   atIndex:0];
        [encoder setBuffer:buffer_slope_ offset:0 atIndex:3];
        [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                    offset:(NSUInteger)output->GetHandle().bytes_offset
                   atIndex:1];

        status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
    } while (0);

    [encoder endEncoding];
    [context_impl commit];
    TNN_PRINT_ENCODER(context_, encoder, this);
    return status;
}

REGISTER_METAL_ACC(PRelu, LAYER_PRELU);

} // namespace TNN_NS
