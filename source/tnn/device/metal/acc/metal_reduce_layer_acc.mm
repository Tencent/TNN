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

#include "tnn/device/metal/acc/metal_reduce_layer_acc.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {
static DimsVector GetKeepDimOutput(const DimsVector& dims_input, ReduceLayerParam *param) {
    DimsVector dims_output(dims_input);
    for(const auto& axis : param->axis) {
        dims_output[axis] = 1;
    }
    return dims_output;
}

MetalReduceLayerAcc::~MetalReduceLayerAcc() {}

Status MetalReduceLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto layer_param     = dynamic_cast<ReduceLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is invalid\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is invalid");
    }

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    for (int i = 0; i < layer_param->axis.size(); ++i) {
        auto axis = layer_param->axis[i];
        need_reformat_ = need_reformat_ || axis == 0 || axis == 1;
    }
    need_reformat_ = need_reformat_ && (layer_param->keep_dims==0);

    if (need_reformat_) {
        dims_output = GetKeepDimOutput(dims_input, layer_param);
    }

    if (layer_param->axis.size() == 1) {
        int axis = layer_param->axis[0];
        multi_axis_ = false;
        axis_ = axis;
        // buffer_param_
        {
            MetalReduceParams metal_params;
            SetDefaultMetalParams(metal_params, dims_input, dims_output);
            FixDefaultMetalParams(metal_params, dims_input, dims_output);
            metal_params.input_batch = dims_input[0];
            metal_params.input_channel = dims_input[1];
            metal_params.output_batch = dims_output[0];
            metal_params.axis  = axis;
            metal_params.input_channel_mode_4 = dims_input[1] % 4;
            buffer_param_                     = [device newBufferWithBytes:(const void *)(&metal_params)
                                                                    length:sizeof(MetalReduceParams)
                                                                   options:MTLResourceCPUCacheModeWriteCombined];
        }
    } else {
        multi_axis_ = true;
        // buffer_param_
        {
            MetalMultiAxisReduceParams metal_params;
            SetDefaultMetalParams(metal_params, dims_input, dims_output);
            FixDefaultMetalParams(metal_params, dims_input, dims_output);
            metal_params.input_batch = dims_input[0];
            metal_params.input_channel = dims_input[1];
            metal_params.output_batch = dims_output[0];
            metal_params.input_channel_mode_4 = dims_input[1] % 4;

            int reduce_length = 1;
            for (auto axis : layer_param->axis) {
                metal_params.reduce_flag[axis] = 1;
                reduce_length *= dims_input[axis];
            }
            metal_params.reduce_length = reduce_length;
            buffer_param_                     = [device newBufferWithBytes:(const void *)(&metal_params)
                                                                    length:sizeof(MetalMultiAxisReduceParams)
                                                                   options:MTLResourceCPUCacheModeWriteCombined];
        }
    }

    if (need_reformat_) {
        MetalSqueezeParams metal_params;
        auto reformat_dims_input = dims_output;
        auto reformat_dims_output = outputs[0]->GetBlobDesc().dims;

        SetDefaultMetalParams(metal_params, reformat_dims_input, reformat_dims_output);
        metal_params.input_channel  = reformat_dims_input[1];
        metal_params.output_channel = reformat_dims_output[1];
        metal_params.input_batch    = reformat_dims_input[0];
        buffer_reformat_   = [device newBufferWithBytes:(const void *)(&metal_params)
                                                length:sizeof(metal_params)
                                                options:MTLResourceCPUCacheModeWriteCombined];
        
        auto data_type_byte_size = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);
        auto buffer_bytes = data_type_byte_size * DimsFunctionUtils::GetDimProduct(reformat_dims_input, 2) * \
                            reformat_dims_input[0] * ROUND_UP(reformat_dims_input[1], 4);
        buffer_output_ = [device newBufferWithLength:buffer_bytes
                                             options:MTLResourceStorageModePrivate];
    }

    return TNN_OK;
}

std::string MetalReduceLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "";
}

Status MetalReduceLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<ReduceLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is invalid\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is invalid");
    }

    auto context_impl = context_->getMetalContextImpl();
    auto encoder      = [context_impl encoder];
    encoder.label = GetKernelLabel();

    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims_output   = output->GetBlobDesc().dims;
    if (need_reformat_) {
        dims_output = GetKeepDimOutput(input->GetBlobDesc().dims, layer_param);
    }

    MetalBandwidth bandwidth;
    Status status        = TNN_OK;
    DataType data_type   = output->GetBlobDesc().data_type;
    string data_type_str = DataTypeUtils::GetDataTypeString(data_type);

    do {
        auto kernel_name = KernelName(inputs, outputs);
        if (kernel_name.length() <= 0) {
            LOGE("Error: empty kernel name\n");
            status = Status(TNNERR_LAYER_ERR, "empty kernel name");
            break;
        }

        status = [context_impl load:[NSString stringWithUTF8String:kernel_name.c_str()]
                            encoder:encoder
                            bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);

        auto output_width  = DimsFunctionUtils::GetDimProduct(dims_output, 3);
        auto output_height = DimsFunctionUtils::GetDim(dims_output, 2);
        auto output_slice = UP_DIV(dims_output[1], 4);
        auto batch         = dims_output[0];
        MTLSize threads = {(NSUInteger)output_width * output_height, (NSUInteger)output_slice, (NSUInteger)batch};

        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->GetHandle().base
                    offset:(NSUInteger)(NSUInteger)input->GetHandle().bytes_offset
                   atIndex:0];
        if (need_reformat_) {
            [encoder setBuffer:buffer_output_
                        offset:0
                       atIndex:1];
        } else {
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                        offset:(NSUInteger)(NSUInteger)output->GetHandle().bytes_offset
                       atIndex:1];
        }
        [encoder setBuffer:buffer_param_ offset:0 atIndex:2];

        status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
        
        if (need_reformat_) {
            threads = GetDefaultThreadSize(outputs[0]->GetBlobDesc().dims, true);
            status = [context_impl load: @"squeeze_common"
                            encoder:encoder
                            bandwidth:bandwidth];
            [encoder setBuffer:buffer_output_
                        offset:0
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                        offset:(NSUInteger)(NSUInteger)output->GetHandle().bytes_offset
                       atIndex:1];
            [encoder setBuffer:buffer_reformat_ offset:0 atIndex:2];
            
            status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
        }
        
    } while (0);

    [encoder endEncoding];
    [context_impl commit];
    TNN_PRINT_ENCODER(context_, encoder, this);
    return status;
}
} // namespace TNN_NS
