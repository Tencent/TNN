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

namespace TNN_NS {

// @brief conv layer metal acc
class MetalBatchNormLayerAcc : public MetalLayerAcc {
public:
    virtual Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    id<MTLBuffer> buffer_scale_ = nil;
    id<MTLBuffer> buffer_bias_  = nil;
};

class MetalScaleLayerAcc : public MetalBatchNormLayerAcc {
public:
    Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
};

Status MetalBatchNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalBatchNormLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                   const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;

    auto layer_res = dynamic_cast<BatchNormLayerResource *>(resource_);
    if (!layer_res) {
        return Status(TNNERR_MODEL_ERR, "Error: layer resource is nil");
    }

    Status status = TNN_OK;
    // buffer_param_
    {
        auto metal_params          = GetDefaultMetalParams(dims_input, dims_output);
        metal_params.share_channel = (layer_res->scale_handle.GetDataCount() == 1) ? 1 : 0;
        buffer_param_              = [device newBufferWithBytes:(const void *)(&metal_params)
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
        RawBuffer raw_buffer = layer_res->bias_handle;
        if (raw_buffer.force_to<float *>() == nullptr) {
            auto buffer               = layer_res->scale_handle;
            const DataType data_type  = buffer.GetDataType();
            const int total_byte_size = dims_output[1] * DataTypeUtils::GetBytesSize(data_type);
            raw_buffer                = RawBuffer(total_byte_size);
        }
        buffer_bias_ = AllocateMetalBufferFormRawBuffer1D(raw_buffer, dims_output[1], status);
    }
    return status;
}

Status MetalBatchNormLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto context_impl = context_->getMetalContextImpl();
    if (!context_impl) {
        LOGE("context_impl is nil\n");
        return Status(TNNERR_CONTEXT_ERR, "MetalBatchNormLayerAcc context_impl is nil");
    }

    auto encoder = [context_impl encoder];
    if (!encoder) {
        LOGE("encoder is nil\n");
        return Status(TNNERR_CONTEXT_ERR, "MetalBatchNormLayerAcc encoder is nil");
    }

    encoder.label = GetKernelLabel();

    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_output  = output->GetBlobDesc().dims;
    auto output_width = dims_output[3], output_height = dims_output[2];
    auto output_slice = UP_DIV(dims_output[1], 4);
    auto batch        = dims_output[0];

    Status status = TNN_OK;
    MetalBandwidth bandwidth;

    do {
        status = [context_impl load:@"batch_norm" encoder:encoder bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);

        MTLSize threads = {(NSUInteger)output_height * output_width, (NSUInteger)output_slice, (NSUInteger)batch};

        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                    offset:(NSUInteger)input->GetHandle().bytes_offset
                   atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                    offset:(NSUInteger)output->GetHandle().bytes_offset
                   atIndex:1];
        [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
        [encoder setBuffer:buffer_scale_ offset:0 atIndex:3];
        [encoder setBuffer:buffer_bias_ offset:0 atIndex:4];

        status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
    } while (0);

    [encoder endEncoding];
    [context_impl commit];
    TNN_PRINT_ENCODER(context_, encoder, this);
    return status;
}

Status MetalScaleLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalBatchNormLayerAcc::AllocateBufferParam(inputs, outputs);
}

REGISTER_METAL_ACC(Scale, LAYER_SCALE);
REGISTER_METAL_ACC(BatchNorm, LAYER_BATCH_NORM);

} // namespace TNN_NS
