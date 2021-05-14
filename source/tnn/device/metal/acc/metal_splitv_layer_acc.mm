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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_METAL_ACC(SplitV, LAYER_SPLITV);

Status MetalSplitVLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalSplitVLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<SplitVLayerParam *>(param_);
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    input_dims[1] = UP_DIV(input_dims[1], 4);
    {
    MetalSplitVParamV2 metal_params;
    metal_params.outer_size = DimsVectorUtils::Count(input_dims, 0, layer_param->axis);
    metal_params.inner_size = DimsFunctionUtils::GetDimProduct(input_dims, layer_param->axis+1);
    metal_params.axis_size  = DimsFunctionUtils::GetDim(input_dims, layer_param->axis);

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalSoftmaxParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    return  TNN_OK;
}

std::string MetalSplitVLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<SplitVLayerParam *>(param_);
    if (layer_param->axis == 1)
        return "splitv_axis_1_common";

    return "splitv_common";
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
    return TNN_OK;
}

Status MetalSplitVLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<SplitVLayerParam *>(param_);
    auto context_impl = context_->getMetalContextImpl();

    auto input = inputs[0];

    MetalBandwidth bandwidth;

    DataType data_type   = input->GetBlobDesc().data_type;
    string data_type_str = DataTypeUtils::GetDataTypeString(data_type);
    if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
        LOGE("MetalSplitVLayerAcc: DataType must be float or half\n");
        return Status(TNNERR_LAYER_ERR, "MetalSplitVLayerAcc: DataType must be float or half");
    }

    const string kernel_name = KernelName(inputs, outputs);
    bool split_channel       = layer_param->axis == 1;

    int axis_offset = 0;
    Status status = TNN_OK;

    for (int i = 0; i < outputs.size(); i++) {
        auto dims_output    = outputs[i]->GetBlobDesc().dims;
        auto output_slice   = UP_DIV(dims_output[1], 4);
        auto split_axis_size= DimsFunctionUtils::GetDim(dims_output, layer_param->axis); 
        split_axis_size = split_channel? output_slice : split_axis_size;

        auto encoder = [context_impl encoder];
        encoder.label = GetKernelLabel();

        do {
            status = [context_impl load: [NSString stringWithUTF8String:kernel_name.c_str()]
                                encoder:encoder
                              bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
            MTLSize threads;
            GetSingleAxisSplitSize(outputs[i]->GetBlobDesc().dims, layer_param->axis, threads, false);

            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                        offset:(NSUInteger)input->GetHandle().bytes_offset
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)outputs[i]->GetHandle().base
                        offset:(NSUInteger)outputs[i]->GetHandle().bytes_offset
                       atIndex:1];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
            [encoder setBytes:&axis_offset length:sizeof(axis_offset) atIndex:3];
            [encoder setBytes:&split_axis_size length:sizeof(split_axis_size) atIndex:4];
            status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
        } while (0);
        [encoder endEncoding];
        [context_impl commit];
        TNN_PRINT_ENCODER(context_, encoder, this);
        BREAK_IF(status != TNN_OK);
        axis_offset += dims_output[layer_param->axis];
    }

    return TNN_OK;
}

REGISTER_METAL_ACC(SplitV, LAYER_SPLITV);
REGISTER_METAL_LAYOUT(LAYER_SPLITV, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
