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

class MetalGroupNormLayerAcc : public MetalLayerAcc {
public:
    Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type);
    virtual Status ConfigBuffer2MetalBlobDesc(BlobDesc &desc);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
};

Status MetalGroupNormLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalGroupNormLayerAcc::ConfigBuffer2MetalBlobDesc(BlobDesc &desc) {
    desc.data_format = DATA_FORMAT_NCHW;
    return TNN_OK;
}

Status MetalGroupNormLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                      const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    auto layer_param = dynamic_cast<GroupNormLayerParam *>(param_);

    Status status = TNN_OK;
    // buffer_param_
    {
        MetalGroupNormParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        FixDefaultMetalParams(metal_params, dims_input, dims_output);

        const int group = layer_param->group;
        const int batch_time_group = DimsFunctionUtils::GetDim(outputs[0]->GetBlobDesc().dims, 0) * group;
        const int channels_per_group = DimsFunctionUtils::GetDim(outputs[0]->GetBlobDesc().dims, 1) / group;
        const int channel_area = DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims, 2);
        const int channel = DimsFunctionUtils::GetDim(outputs[0]->GetBlobDesc().dims, 1);
        const int group_area = channel_area * channels_per_group;
        
        if (0 == group_area || 0 == channels_per_group) {
            LOGE("Error: blob count is zero\n");
            return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
        }
        metal_params.group = group;
        metal_params.channel = channel;
        metal_params.channel_area = channel_area;
        metal_params.group_area   = group_area;
        metal_params.batch_time_group = batch_time_group;
        metal_params.channels_per_group = channels_per_group;
        metal_params.eps          = layer_param->eps;
        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(metal_params)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    return status;
}

std::vector<DataFormat> MetalGroupNormLayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    return {DATA_FORMAT_NCHW};
}

Status MetalGroupNormLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto context_impl = context_->getMetalContextImpl();

    auto layer_param = dynamic_cast<GroupNormLayerParam *>(param_);

    auto input_blob  = inputs[0];
    auto scale_blob  = inputs[1];
    auto bias_blob   = inputs[2];
    auto output_blob = outputs[0];
    auto dims_input  = input_blob->GetBlobDesc().dims;

    const int group = layer_param->group;
    const int batch_time_group = DimsFunctionUtils::GetDim(outputs[0]->GetBlobDesc().dims, 0) * group;
    const int channels_per_group = DimsFunctionUtils::GetDim(outputs[0]->GetBlobDesc().dims, 1) / group;
    const int channel_area = DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims, 2);
    const int group_area = channel_area * channels_per_group;
    
    if (0 == group_area || 0 == channels_per_group) {
        LOGE("Error: blob count is zero\n");
        return Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
    }

    Status status = TNN_OK;
    MetalBandwidth bandwidth;
    {
        auto encoder = [context_impl encoder];
        encoder.label = GetKernelLabel();

        do {
            status = [context_impl load:@"group_norm" encoder:encoder bandwidth:bandwidth];
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

            [encoder dispatchThreadgroups:{(NSUInteger)1, (NSUInteger)1, (NSUInteger)batch_time_group}
                    threadsPerThreadgroup:{(NSUInteger)32, (NSUInteger)1, (NSUInteger)1}];
            BREAK_IF(status != TNN_OK);
        } while (0);

        [encoder endEncoding];
        [context_impl commit];
        TNN_PRINT_ENCODER(context_, encoder, this);
    }

    return status;
}

REGISTER_METAL_ACC(GroupNorm, LAYER_GROUP_NORM);
REGISTER_METAL_LAYOUT(LAYER_GROUP_NORM, DATA_FORMAT_NCHW);

} // namespace TNN_NS
