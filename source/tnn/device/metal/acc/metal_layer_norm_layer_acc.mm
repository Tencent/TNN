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

class MetalLayerNormLayerAcc : public MetalLayerAcc {
public:
    Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type);
    virtual Status ConfigBuffer2MetalBlobDesc(BlobDesc &desc);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
};

Status MetalLayerNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalLayerNormLayerAcc::ConfigBuffer2MetalBlobDesc(BlobDesc &desc) {
    desc.data_format = DATA_FORMAT_NCHW;
    return TNN_OK;
}

Status MetalLayerNormLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                      const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    auto layer_param = dynamic_cast<LayerNormLayerParam *>(param_);

    Status status = TNN_OK;
    // buffer_param_
    {
        MetalLayerNormParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        FixDefaultMetalParams(metal_params, dims_input, dims_output);

        const int reduce_dim_size = layer_param->reduce_dims_size;
        const int channel_dim_size = (int)dims_input.size() - reduce_dim_size;

        const int channels = DimsVectorUtils::Count(dims_input, 0, channel_dim_size);
        const int channel_area = DimsVectorUtils::Count(dims_output, channel_dim_size);
        if (0 == channels || 0 == channel_area) {
            LOGE("Error: blob count is zero\n");
            return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
        }
        metal_params.channel_area = channel_area;
        metal_params.channels     = channels;
        metal_params.eps          = layer_param->eps;
        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(metal_params)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    return status;
}

std::vector<DataFormat> MetalLayerNormLayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    return {DATA_FORMAT_NCHW};
}

Status MetalLayerNormLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto context_impl = context_->getMetalContextImpl();

    auto layer_param = dynamic_cast<LayerNormLayerParam *>(param_);

    auto input_blob  = inputs[0];
    auto scale_blob  = inputs[1];
    auto bias_blob   = inputs[2];
    auto output_blob = outputs[0];
    auto dims_input  = input_blob->GetBlobDesc().dims;

    const int reduce_dim_size  = layer_param->reduce_dims_size;
    const int channel_dim_size = static_cast<int>(dims_input.size() - reduce_dim_size);

    const int channels = DimsVectorUtils::Count(dims_input, 0, channel_dim_size);
    const int channel_area = DimsVectorUtils::Count(output_blob->GetBlobDesc().dims, channel_dim_size);

    if (0 == channels || 0 == channel_area) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    Status status = TNN_OK;
    MetalBandwidth bandwidth;
    {
        auto encoder = [context_impl encoder];
        encoder.label = GetKernelLabel();

        do {
            status = [context_impl load:@"layer_norm" encoder:encoder bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);

            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input_blob->GetHandle().base
                        offset:(NSUInteger)input_blob->GetHandle().bytes_offset
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)scale_blob->GetHandle().base
                        offset:(NSUInteger)scale_blob->GetHandle().bytes_offset
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)bias_blob->GetHandle().base
                        offset:(NSUInteger)bias_blob->GetHandle().bytes_offset
                       atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output_blob->GetHandle().base
                        offset:(NSUInteger)output_blob->GetHandle().bytes_offset
                       atIndex:3];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:4];

            [encoder dispatchThreadgroups:{(NSUInteger)1, (NSUInteger)1, (NSUInteger)channels}
                    threadsPerThreadgroup:{(NSUInteger)32, (NSUInteger)1, (NSUInteger)1}];
            BREAK_IF(status != TNN_OK);
        } while (0);

        [encoder endEncoding];
        [context_impl commit];
        TNN_PRINT_ENCODER(context_, encoder, this);
    }

    return status;
}

REGISTER_METAL_ACC(LayerNorm, LAYER_LAYER_NORM);
REGISTER_METAL_LAYOUT(LAYER_LAYER_NORM, DATA_FORMAT_NCHW);

} // namespace TNN_NS
