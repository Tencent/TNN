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

#include "tnn/device/metal/acc/metal_multidir_broadcast_layer_acc.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils.h"

namespace TNN_NS {
MetalMultidirBroadcastLayerAcc::~MetalMultidirBroadcastLayerAcc() {}

Status MetalMultidirBroadcastLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                           const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: layer param is nil");
    }
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalBroadcastParams metal_params;
        metal_params.input_width  = dims_output[3];
        metal_params.input_height = dims_output[2];
        metal_params.input_size   = metal_params.input_height * metal_params.input_width;
        metal_params.input_slice  = UP_DIV(dims_output[1], 4);

        metal_params.output_width  = dims_output[3];
        metal_params.output_height = dims_output[2];
        metal_params.output_size   = metal_params.output_height * metal_params.output_width;
        metal_params.output_slice  = UP_DIV(dims_output[1], 4);

        metal_params.batch = dims_output[0];

        metal_params.broadcast_input0 = layer_param->input0_broadcast_type;
        metal_params.broadcast_input1 = layer_param->input1_broadcast_type;
        ;

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalBroadcastParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource_);

    Status status = TNN_OK;
    if (layer_res && !buffer_weight_) {
        const auto element_shape = layer_res->element_shape;
        buffer_weight_ =
            AllocatePackedNC4HW4MetalBufferFormRawBuffer(layer_res->element_handle, element_shape, 1, status);
    }
    return status;
}

Status MetalMultidirBroadcastLayerAcc::SetKernelEncoderParam(
                                                             id<MTLComputeCommandEncoder> encoder,
                                                             const std::vector<Blob *> &inputs,
                                                             const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    
    auto input0 = inputs[0];
    auto output = outputs[0];
    
    auto input_buffer0 = (__bridge id<MTLBuffer>)(void *)input0->GetHandle().base;
    auto input_buffer0_bytes_offset = (NSUInteger)input0->GetHandle().bytes_offset;

    auto input_buffer1 = buffer_weight_;
    auto input_buffer1_bytes_offset = (NSUInteger)0;
    
    if (buffer_weight_) {
        if (layer_param->weight_input_index == 0) {
            std::swap(input_buffer0, input_buffer1);
            std::swap(input_buffer0_bytes_offset, input_buffer1_bytes_offset);
        }
    } else {
        if (inputs.size() <= 1) {
            input_buffer1              = input_buffer0;
            input_buffer1_bytes_offset = input_buffer0_bytes_offset;
        } else {
            input_buffer1              = (__bridge id<MTLBuffer>)(void *)inputs[1]->GetHandle().base;
            input_buffer1_bytes_offset = (NSUInteger)inputs[1]->GetHandle().bytes_offset;
        }
    }
    
    [encoder setBuffer:input_buffer0 offset:input_buffer0_bytes_offset atIndex:0];
    [encoder setBuffer:input_buffer1 offset:input_buffer1_bytes_offset atIndex:1];

    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                offset:(NSUInteger)output->GetHandle().bytes_offset
               atIndex:2];
    [encoder setBuffer:buffer_param_ offset:0 atIndex:3];
    return TNN_OK;
}

Status MetalMultidirBroadcastLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                               const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: layer param is nil");
    }

    if (!((inputs.size() == 1 && buffer_weight_) || inputs.size() == 2)) {
        LOGE("Error: MetalMultidirBroadcastLayerAcc invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "MetalMultidirBroadcastLayerAcc invalid inputs count");
    }
    
    return MetalLayerAcc::Forward(inputs, outputs);
}
} // namespace TNN_NS
