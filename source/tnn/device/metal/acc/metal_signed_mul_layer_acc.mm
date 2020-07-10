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
#include "tnn/interpreter/layer_param.h"

namespace TNN_NS {
DECLARE_METAL_UNARY_ACC(SignedMul, LAYER_SIGNED_MUL);

string MetalSignedMulLayerAcc::KernelName() {
    return "signed_mul";
}

Status MetalSignedMulLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<SignedMulLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: SignedMulLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "SignedMulLayerParam is nil");
    }

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    if (buffer_param_ == nil) {
        MetalSignedMulParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);

        metal_params.alpha = layer_param->alpha;
        metal_params.beta  = layer_param->beta;
        if (layer_param->gamma == 0) {
            LOGE("Error: SignedMulLayerParam.gamma should not be 0\n");
            return Status(TNNERR_MODEL_ERR, "SignedMulLayerParam gamma=0");
        }
        metal_params.gamma_inv = 1.0 / layer_param->gamma;

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalSignedMulParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalSignedMulLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto context_impl = context_->getMetalContextImpl();
    auto encoder      = [context_impl encoder];
    if (param_) {
        encoder.label = [NSString stringWithFormat:@"layer: %s ", param_->name.c_str()];
    }

    MTLSize threads;
    auto status = ComputeThreadSize(inputs, outputs, threads);
    if (status != TNN_OK) {
        return status;
    }

    MetalBandwidth bandwidth;
    do {
        if (inputs[0]->GetBlobDesc().dims[1] == 4) {
            status = [context_impl load:@"signed_mul_fused_channel4" encoder:encoder bandwidth:bandwidth];
        } else {
            status = [context_impl load:@"signed_mul_fused" encoder:encoder bandwidth:bandwidth];
        }

        BREAK_IF(status != TNN_OK);
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                    offset:(NSUInteger)input->GetHandle().bytes_offset
                   atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                    offset:(NSUInteger)output->GetHandle().bytes_offset
                   atIndex:1];
        [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
        status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
    } while (0);

    [encoder endEncoding];
    [context_impl commit];
    TNN_PRINT_ENCODER(context_, encoder, this);
    return status;
}

REGISTER_METAL_UNARY_ACC(SignedMul, LAYER_SIGNED_MUL);

} // namespace TNN_NS