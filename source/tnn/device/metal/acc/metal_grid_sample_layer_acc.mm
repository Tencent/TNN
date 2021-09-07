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
#include "tnn/device/metal/acc/metal_unary_layer_acc.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
DECLARE_METAL_ACC_WITH_EXTRA(GridSample, LAYER_GRIDSAMPLE, protected: virtual bool UseNaiveConstantBlobs(){return true;});

Status MetalGridSampleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalGridSampleLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                    const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto dims_input      = inputs[0]->GetBlobDesc().dims;
    auto dims_input1     = inputs[1]->GetBlobDesc().dims;
    auto dims_output     = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalParams metal_params;
        metal_params.input_width  = dims_input[3];
        metal_params.input_height = dims_input[2];
        metal_params.input_size   = metal_params.input_height * metal_params.input_width;
        metal_params.input_slice  = UP_DIV(dims_input[1], 4);
        // metal_params.input_channel      = dims_input[1];

        metal_params.output_width  = dims_output[3];
        metal_params.output_height = dims_output[2];
        metal_params.output_size   = metal_params.output_height * metal_params.output_width;
        metal_params.output_slice  = UP_DIV(dims_output[1], 4);
        // metal_params.output_channel     = dims_output[1];
        metal_params.batch = dims_output[0];

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(metal_params)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

string MetalGridSampleLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    return "grid_sample";
}

Status MetalGridSampleLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                                  MTLSize &size) {

    return MetalLayerAcc::ComputeThreadSize(inputs, outputs, size);
}

Status MetalGridSampleLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

Status MetalGridSampleLayerAcc::SetKernelEncoderParam(id<MTLComputeCommandEncoder> encoder,
                                                      const std::vector<Blob *> &inputs,
                                                      const std::vector<Blob *> &outputs) {
    int bytes_offset_mat_a = 0;
    int bytes_offset_mat_b = 0;
    id<MTLBuffer> matrix_a = nil;
    id<MTLBuffer> matrix_b = nil;
    matrix_a               = (__bridge id<MTLBuffer>)(inputs[0]->GetHandle().base);
    matrix_b               = (__bridge id<MTLBuffer>)(inputs[1]->GetHandle().base);
    bytes_offset_mat_a     = inputs[0]->GetHandle().bytes_offset;
    bytes_offset_mat_b     = inputs[1]->GetHandle().bytes_offset;

    [encoder setBuffer:matrix_a offset:(NSUInteger)bytes_offset_mat_a atIndex:0];
    [encoder setBuffer:matrix_b offset:(NSUInteger)bytes_offset_mat_b atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)outputs[0]->GetHandle().base
                offset:(NSUInteger)outputs[0]->GetHandle().bytes_offset
               atIndex:2];
    [encoder setBuffer:buffer_param_ offset:0 atIndex:3];

    return TNN_OK;
}

REGISTER_METAL_ACC(GridSample, LAYER_GRIDSAMPLE);
REGISTER_METAL_LAYOUT(LAYER_GRIDSAMPLE, DATA_FORMAT_NC4HW4);
} // namespace TNN_NS
