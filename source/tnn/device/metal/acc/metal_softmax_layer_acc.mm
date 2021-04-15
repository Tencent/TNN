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

DECLARE_METAL_ACC(Softmax, LAYER_SOFTMAX);

Status MetalSoftmaxLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<SoftmaxLayerParam *>(param_);
    const auto& input_dims = inputs[0]->GetBlobDesc().dims;
    layer_param->axis = (layer_param->axis + input_dims.size()) % input_dims.size();

    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalSoftmaxLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    auto layer_param = dynamic_cast<SoftmaxLayerParam *>(param_);
    if (layer_param->axis != 1) {
        auto dims_input  = inputs[0]->GetBlobDesc().dims;
        auto input_channel = dims_input[1];
        dims_input[1] = UP_DIV(input_channel, 4);
        MetalArgMaxOrMinParams metal_params;
        metal_params.input_channel = input_channel;
        auto axis = layer_param->axis;
        metal_params.reduce_size = dims_input[axis];
        metal_params.outer_size  = DimsVectorUtils::Count(dims_input, 0, axis);
        metal_params.inner_size  = DimsVectorUtils::Count(dims_input, axis+1);

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalArgMaxOrMinParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
        return TNN_OK;
    }
    // buffer_param_
    {
        MetalSoftmaxParams metal_params;
        metal_params.output_width   = DimsFunctionUtils::GetDimProduct(output_dims, 3);
        metal_params.output_height  = DimsFunctionUtils::GetDim(output_dims, 2);
        metal_params.output_size    = metal_params.output_height * metal_params.output_width;
        metal_params.output_slice   = UP_DIV(output_dims[1], 4);
        metal_params.channel_remain = output_dims[1] % 4;

        metal_params.batch = output_dims[0];

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalSoftmaxParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

std::string MetalSoftmaxLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "";
}

Status MetalSoftmaxLayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

Status MetalSoftmaxLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    return MetalLayerAcc::ComputeThreadSize(inputs, outputs, size);
}

Status MetalSoftmaxLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<SoftmaxLayerParam *>(param_);
    auto context_impl              = context_->getMetalContextImpl();
    auto encoder                   = [context_impl encoder];
    encoder.label = GetKernelLabel();

    auto input  = inputs[0];
    auto output = outputs[0];

    MetalBandwidth bandwidth;
    Status status      = TNN_OK;
    DataType data_type = output->GetBlobDesc().data_type;

    do {
        if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
            LOGE("data type(%d) is unsupported\n", data_type);
            status = Status(TNNERR_LAYER_ERR, "data type is unsupported");
        }
        BREAK_IF(status != TNN_OK);
        auto output_dims    = output->GetBlobDesc().dims;
        auto batch          = output_dims[0];
        auto output_channel = output_dims[1];
        auto output_height  = DimsFunctionUtils::GetDim(output_dims, 2);
        auto output_width   = DimsFunctionUtils::GetDimProduct(output_dims, 3);
        auto output_slice   = UP_DIV(output_dims[1], 4);
        auto mode           = output_dims[1] % 4;

        MTLSize threads;
        if (layer_param->axis == 1) {
            threads = {(NSUInteger)output_width * output_height, 1, (NSUInteger)batch};
            if (output_slice == 1) {
                status =
                    [context_impl load:[NSString stringWithFormat:@"softmax_axis_1_slice_1_channel_%d", output_channel]
                               encoder:encoder
                             bandwidth:bandwidth];
            } else {
                if (mode == 0) {
                    status = [context_impl load:@"softmax_axis_1_common_mode_0" encoder:encoder bandwidth:bandwidth];
                } else {
                    status = [context_impl load:@"softmax_axis_1_common" encoder:encoder bandwidth:bandwidth];
                }
            }
        } else {
            GetSingleAxisSplitSize(output_dims, layer_param->axis, threads, true);
            status  = [context_impl load:@"softmax_common" encoder:encoder bandwidth:bandwidth];
        }
        BREAK_IF(status != TNN_OK);

        status = SetKernelEncoderParam(encoder, inputs, outputs);
        BREAK_IF(status != TNN_OK);

        status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
    } while (0);

    [encoder endEncoding];
    [context_impl commit];
    TNN_PRINT_ENCODER(context_, encoder, this);
    return status;
}

REGISTER_METAL_ACC(Softmax, LAYER_SOFTMAX);
REGISTER_METAL_LAYOUT(LAYER_SOFTMAX, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
