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
#include "tnn/utils/half_utils.h"

namespace TNN_NS {

// @brief conv layer metal acc
class MetalInstanceNormLayerAcc : public MetalLayerAcc {
public:
    Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    id<MTLBuffer> buffer_scale_final_ = nil;
    id<MTLBuffer> buffer_bias_final_  = nil;
protected:
    id<MTLBuffer> buffer_scale_ = nil;
    id<MTLBuffer> buffer_bias_  = nil;
};

Status MetalInstanceNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalInstanceNormLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                      const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    auto layer_res = dynamic_cast<InstanceNormLayerResource *>(resource_);
    if (!layer_res) {
        return Status(TNNERR_MODEL_ERR, "Error: layer resource is nil");
    }

    Status status = TNN_OK;
    // buffer_param_
    {
        auto metal_params = GetDefaultMetalParams(dims_input, dims_output);
        // adapt to batchnorm, merge output batch to slice
        metal_params.output_slice *= metal_params.batch;
        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(metal_params)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    if (!buffer_scale_) {
        buffer_scale_ = AllocateMetalBufferFormRawBuffer1D(layer_res->scale_handle, dims_output[1], status);
        if (status != TNN_OK) {
            return status;
        }
    }

    if (!buffer_bias_) {
        buffer_bias_ = AllocateMetalBufferFormRawBuffer1D(layer_res->bias_handle, dims_output[1], status);
        if (status != TNN_OK) {
            return status;
        }
    }

    const int total_channel = dims_input[0] * ROUND_UP(dims_input[1], 4);
    auto zero_buffer        = RawBuffer(total_channel);
    if (!buffer_scale_final_) {
        buffer_scale_final_ = AllocateMetalBufferFormRawBuffer1D(zero_buffer, total_channel, status);
        if (status != TNN_OK) {
            return status;
        }
    }

    if (!buffer_bias_final_) {
        buffer_bias_final_ = AllocateMetalBufferFormRawBuffer1D(zero_buffer, total_channel, status);
        if (status != TNN_OK) {
            return status;
        }
    }
    return status;
}

Status MetalInstanceNormLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto context_impl = context_->getMetalContextImpl();
    if (!context_impl) {
        LOGE("context_impl is nil\n");
        return Status(TNNERR_CONTEXT_ERR, "MetalInstanceNormLayerAcc context_impl is nil");
    }

    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_output  = output->GetBlobDesc().dims;
    auto output_slice = UP_DIV(dims_output[1], 4);
    auto batch        = dims_output[0];

    Status status = TNN_OK;
    MetalBandwidth bandwidth;

    // stage 1, update scale and bias
    {
        auto encoder = [context_impl encoder];
        if (!encoder) {
            LOGE("encoder is nil\n");
            return Status(TNNERR_CONTEXT_ERR, "instance_norm_var_bias encoder is nil");
        }

        encoder.label = GetKernelLabel();

        do {
            status = [context_impl load:@"instance_norm" encoder:encoder bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);

            LOGD("bandwidth.thread_execution_width: %d\n", (int)bandwidth.thread_execution_width);
            LOGD("bandwidth.max_threads_per_group: %d\n", (int)bandwidth.max_threads_per_group);

            [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
            [encoder setBuffer:buffer_scale_ offset:0 atIndex:3];
            [encoder setBuffer:buffer_bias_ offset:0 atIndex:4];
            
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                        offset:(NSUInteger)output->GetHandle().bytes_offset
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                        offset:(NSUInteger)input->GetHandle().bytes_offset
                       atIndex:0];

            // do not use setThreadgroupMemoryLength, unknown bug will raise
            //            [encoder setThreadgroupMemoryLength:8*8*4*data_type_size atIndex:0];
            [encoder dispatchThreadgroups:{(NSUInteger)1, (NSUInteger)1, (NSUInteger)batch * output_slice}
                    threadsPerThreadgroup:{(NSUInteger)32, (NSUInteger)1, (NSUInteger)1}];
            BREAK_IF(status != TNN_OK);
        } while (0);

        [encoder endEncoding];
        [context_impl commit];
        TNN_PRINT_ENCODER(context_, encoder, this);
    }

    return status;
}

REGISTER_METAL_ACC(InstanceNorm, LAYER_INST_BATCH_NORM);

} // namespace TNN_NS
