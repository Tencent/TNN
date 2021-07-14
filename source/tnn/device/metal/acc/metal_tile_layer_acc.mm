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

namespace TNN_NS {
    DECLARE_METAL_ACC(Tile, LAYER_REPEAT);

    Status MetalTileLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
        return MetalLayerAcc::Reshape(inputs, outputs);
    }

    Status MetalTileLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                  const std::vector<Blob *> &outputs) {
        id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
        auto dims_input      = inputs[0]->GetBlobDesc().dims;
        auto dims_output     = outputs[0]->GetBlobDesc().dims;
        // buffer_param_
        {
            MetalTileParams metal_params;
            if(dims_input.size() == 4) {
                metal_params.input_dim0_size = 1;
                metal_params.input_dim1_size = 1;
                metal_params.output_dim1_size     = 1;
                metal_params.output_dim0_size     = 1;
                metal_params.extend_dim1_times    = 1;
                metal_params.extend_dim0_times    = 1;
            }
            if(dims_input.size() == 5) {
                metal_params.input_dim0_size = 1;
                metal_params.input_dim1_size = dims_input[4];
                metal_params.output_dim1_size     = dims_output[4];
                metal_params.output_dim0_size     = 1;
                metal_params.extend_dim1_times    = dims_output[4] / dims_input[4];
                metal_params.extend_dim0_times    = 1;
            }

            if(dims_input.size() == 6) {
                metal_params.input_dim0_size = dims_input[5];
                metal_params.input_dim1_size = dims_input[4];
                metal_params.output_dim0_size     = dims_output[5];
                metal_params.output_dim1_size     = dims_output[4];
                metal_params.extend_dim0_times    = dims_output[5] / dims_input[5];
                metal_params.extend_dim1_times    = dims_output[4] / dims_input[4];
            }
            metal_params.input_width          = dims_input[3];
            metal_params.input_height         = dims_input[2];
            metal_params.input_size           = metal_params.input_height * metal_params.input_width;
            metal_params.input_slice          = UP_DIV(dims_input[1], 4);
            metal_params.input_channel        = dims_input[1];

            metal_params.output_width         = dims_output[3]*metal_params.output_dim1_size * metal_params.output_dim0_size;
            metal_params.output_height        = dims_output[2];
            metal_params.output_size          = metal_params.output_height * metal_params.output_width;
            metal_params.output_slice         = UP_DIV(dims_output[1], 4);
            metal_params.output_channel       = dims_output[1];
            metal_params.batch                = dims_output[0];
            metal_params.extend_batch_times   = dims_output[0] / dims_input[0];
            metal_params.extend_channel_times = dims_output[1] / dims_input[1];
            metal_params.extend_width_times   = dims_output[3] / dims_input[3];

            LOGE("input_width=%d\n",metal_params.input_width);

            buffer_param_     = [device newBufferWithBytes:(const void *)(&metal_params)
                                                    length:sizeof(metal_params)
                                                   options:MTLResourceCPUCacheModeWriteCombined];
        }
        return TNN_OK;
    }

    string MetalTileLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
        return "tile";
    }

    Status MetalTileLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                                const std::vector<Blob *> &outputs,
                                                MTLSize &size) {
        return MetalLayerAcc::ComputeThreadSize(inputs, outputs, size);
    }

    Status MetalTileLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
        return MetalLayerAcc::Forward(inputs, outputs);
    }

    Status MetalTileLayerAcc::SetKernelEncoderParam(
            id<MTLComputeCommandEncoder> encoder,
            const std::vector<Blob *> &inputs,
            const std::vector<Blob *> &outputs) {
        return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
    }

    REGISTER_METAL_ACC(Tile, LAYER_REPEAT);
    REGISTER_METAL_LAYOUT(LAYER_REPEAT, DATA_FORMAT_NC4HW4);
} // namespace TNN_NS
