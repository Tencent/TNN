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

DECLARE_METAL_ACC(SplitV, LAYER_SPLITV);

Status MetalSplitVLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalSplitVLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return  MetalLayerAcc::AllocateBufferParam(inputs, outputs);
}

std::string MetalSplitVLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "";
}

Status MetalSplitVLayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

Status MetalSplitVLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    return MetalLayerAcc::ComputeThreadSize(inputs, outputs, size);
}

Status MetalSplitVLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<SplitVLayerParam *>(param_);
    if (!layer_param || layer_param->axis != 1) {
        LOGE("SplitV do not support axis!=1 \n");
        return Status(TNNERR_LAYER_ERR, "SplitV axis is not supported");
    }

    auto context_impl = context_->getMetalContextImpl();

    auto input = inputs[0];

    MetalBandwidth bandwidth;
    Status status        = TNN_OK;
    DataType data_type   = input->GetBlobDesc().data_type;
    string data_type_str = DataTypeUtils::GetDataTypeString(data_type);
    if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
        LOGE("MetalSplitVLayerAcc: DataType must be float or half\n");
        return Status(TNNERR_LAYER_ERR, "MetalSplitVLayerAcc: DataType must be float or half");
    }

    int channel_offset = 0;
    for (int i = 0; i < outputs.size(); i++) {
        auto dims_output    = outputs[i]->GetBlobDesc().dims;
        auto output_width   = dims_output[3];
        auto output_height  = dims_output[2];
        auto output_channel = dims_output[1];
        auto output_slice   = UP_DIV(dims_output[1], 4);
        auto batch          = dims_output[0];

        auto encoder = [context_impl encoder];
        encoder.label = GetKernelLabel();

        do {
            status = [context_impl load: @"splitv_axis_1_common"
                                encoder:encoder
                              bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
            MTLSize threads = {(NSUInteger)output_height * output_width, (NSUInteger)output_slice, (NSUInteger)batch};

            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                        offset:(NSUInteger)input->GetHandle().bytes_offset
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)outputs[i]->GetHandle().base
                        offset:(NSUInteger)outputs[i]->GetHandle().bytes_offset
                       atIndex:1];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
            [encoder setBytes:&channel_offset length:sizeof(channel_offset) atIndex:3];
            [encoder setBytes:&output_slice length:sizeof(output_slice) atIndex:4];
            status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
        } while (0);
        [encoder endEncoding];
        [context_impl commit];
        TNN_PRINT_ENCODER(context_, encoder, this);
        BREAK_IF(status != TNN_OK);
        channel_offset += output_channel;
    }
    return TNN_OK;
}

REGISTER_METAL_ACC(SplitV, LAYER_SPLITV);

} // namespace TNN_NS
