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
// @brief concat layer metal acc
class MetalConcatLayerAcc : public MetalLayerAcc {
public:
    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    // @brief virtual destrcutor
    virtual ~MetalConcatLayerAcc();

    Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    NSArray<id<MTLBuffer>> *buffer_input_params_ = nil;
    Status isSupported(LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                       const std::vector<Blob *> &outputs);
};

Status MetalConcatLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = isSupported(param, resource, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    return MetalLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status MetalConcatLayerAcc::isSupported(LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs) {

    auto layer_param = dynamic_cast<ConcatLayerParam *>(param);
    if (!layer_param || (layer_param->axis != 1 && layer_param->axis != 2 && layer_param->axis != 3)) {
        LOGE("Concat do not support axis =%d\n", layer_param->axis);
        return Status(TNNERR_LAYER_ERR, "Concat type is not supported");
    }

    if (inputs.size() < 2) {
        LOGE("Error: Concat's input blob number must >= 2\n");
        return Status(TNNERR_NET_ERR, "Error: Concat's input blob number must >= 2\n");
    } else if (inputs.size() == 2) {
        return TNN_OK;
    }
    return TNN_OK;
}

MetalConcatLayerAcc::~MetalConcatLayerAcc() {}

Status MetalConcatLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = isSupported(param_, resource_, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto dims_output = outputs[0]->GetBlobDesc().dims;
    {
        auto dims_input_0 = inputs[0]->GetBlobDesc().dims;
        auto dims_input_1 = inputs[1]->GetBlobDesc().dims;
        MetalConcatParams metal_params;

        metal_params.input_size      = dims_input_0[2] * dims_input_0[3];
        metal_params.input_channel_0 = dims_input_0[1];
        metal_params.input_slice_0   = UP_DIV(dims_input_0[1], 4);

        metal_params.input_channel_1 = dims_input_1[1];
        metal_params.input_slice_1   = UP_DIV(dims_input_1[1], 4);

        metal_params.output_channel = dims_output[1];
        metal_params.output_size    = dims_output[2] * dims_output[3];
        metal_params.output_slice   = UP_DIV(dims_output[1], 4);

        metal_params.batch = dims_output[0];

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalConcatParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    int channel_offset = 0;
    auto buffer_params = [NSMutableArray array];
    for (int ii = 0; ii < inputs.size(); ii++) {
        auto dims_input_0 = inputs[ii]->GetBlobDesc().dims;
        MetalConcatParams metal_params;
        metal_params.input_width   = dims_input_0[3];
        metal_params.input_height  = dims_input_0[2];
        metal_params.input_size      = dims_input_0[2] * dims_input_0[3];
        metal_params.input_channel_0 = dims_input_0[1];
        metal_params.input_slice_0   = UP_DIV(dims_input_0[1], 4);

        metal_params.input_channel_offset = channel_offset;
        channel_offset += dims_input_0[1];

        metal_params.output_width   = dims_output[3];
        metal_params.output_channel = dims_output[1];
        metal_params.output_size    = dims_output[2] * dims_output[3];
        metal_params.output_slice   = UP_DIV(dims_output[1], 4);

        metal_params.batch = dims_output[0];

        auto param = [device newBufferWithBytes:(const void *)(&metal_params)
                                         length:sizeof(MetalConcatParams)
                                        options:MTLResourceCPUCacheModeWriteCombined];
        if (param) {
            [buffer_params addObject:param];
        } else {
            LOGE("Error: MetalConcatLayerAcc::AllocateBufferParam failed\n");
            return Status(TNNERR_NET_ERR, "etalConcatLayerAcc::AllocateBufferParam failed");
        }
    }
    buffer_input_params_ = buffer_params;

    return TNN_OK;
}

Status MetalConcatLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status = isSupported(param_, resource_, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    auto layer_param = dynamic_cast<ConcatLayerParam *>(param_);

    auto context_impl = context_->getMetalContextImpl();

    auto output = outputs[0];

    auto dims_output   = output->GetBlobDesc().dims;
    auto output_width  = dims_output[3];
    auto output_height = dims_output[2];
    auto output_slice  = UP_DIV(dims_output[1], 4);
    auto batch         = dims_output[0];

    MetalBandwidth bandwidth;

    DataType data_type       = output->GetBlobDesc().data_type;
    string data_type_str     = DataTypeUtils::GetDataTypeString(data_type);
    const int data_type_size = DataTypeUtils::GetBytesSize(data_type);
    
    if (layer_param->axis == 1) {
        if (inputs.size() == 2) {
            auto input_0 = inputs[0];
            auto input_1 = inputs[1];

            auto encoder = [context_impl encoder];
            encoder.label = GetKernelLabel();

            do {
                status = [context_impl load:@"concat_axis_1_common"
                                    encoder:encoder
                                  bandwidth:bandwidth];
                BREAK_IF(status != TNN_OK);
                
                auto threads =  MTLSizeMake(output_width * output_height, output_slice, batch);
                [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input_0->GetHandle().base
                            offset:(NSUInteger)input_0->GetHandle().bytes_offset
                           atIndex:0];
                [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input_1->GetHandle().base
                            offset:(NSUInteger)input_1->GetHandle().bytes_offset
                           atIndex:1];
                [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                            offset:(NSUInteger)output->GetHandle().bytes_offset
                           atIndex:2];
                [encoder setBuffer:buffer_param_ offset:0 atIndex:3];

                status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
                BREAK_IF(status != TNN_OK);
            } while (0);

            [encoder endEncoding];
            [context_impl commit];
            TNN_PRINT_ENCODER(context_, encoder, this);
        } else {
            for (int i = 0; i < inputs.size(); i++) {
                auto dims_input    = inputs[i]->GetBlobDesc().dims;
                auto input_width   = dims_input[3];
                auto input_height  = dims_input[2];
                auto input_channel = dims_input[1];
                auto input_slice   = UP_DIV(input_channel, 4);
                auto batch         = dims_input[0];

                auto encoder = [context_impl encoder];
                encoder.label = GetKernelLabel();

                do {
                    status = [context_impl load:@"concat_axis_1_common_x"
                                        encoder:encoder
                                      bandwidth:bandwidth];
                    BREAK_IF(status != TNN_OK);

                    auto threads =  MTLSizeMake(output_width * output_height, input_slice, batch);

                    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)inputs[i]->GetHandle().base
                                offset:(NSUInteger)inputs[i]->GetHandle().bytes_offset
                               atIndex:0];
                    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                                offset:(NSUInteger)output->GetHandle().bytes_offset
                               atIndex:1];
                    [encoder setBuffer:buffer_input_params_[i] offset:0 atIndex:2];

                    status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
                    BREAK_IF(status != TNN_OK);
                } while (0);

                [encoder endEncoding];
                [context_impl commit];
                TNN_PRINT_ENCODER(context_, encoder, this);
                BREAK_IF(status != TNN_OK);
            }
        }
    } else {
        int offset[2] = {0, 0};
        for (int i = 0; i < inputs.size(); i++) {
            auto dims_input    = inputs[i]->GetBlobDesc().dims;
            auto input_width   = dims_input[3];
            auto input_height  = dims_input[2];
            auto input_channel = dims_input[1];
            auto input_slice   = UP_DIV(input_channel, 4);
            auto batch         = dims_input[0];

            auto encoder = [context_impl encoder];
            encoder.label = GetKernelLabel();
            
            do {
                status = [context_impl load:@"concat_axis_23_common_x"
                                    encoder:encoder
                                  bandwidth:bandwidth];
                BREAK_IF(status != TNN_OK);

                auto threads =  MTLSizeMake(output_width, output_height, input_slice*batch);

                [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)inputs[i]->GetHandle().base
                            offset:(NSUInteger)inputs[i]->GetHandle().bytes_offset
                           atIndex:0];
                [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                            offset:(NSUInteger)output->GetHandle().bytes_offset
                           atIndex:1];
                [encoder setBuffer:buffer_input_params_[i] offset:0 atIndex:2];
                [encoder setBytes:offset length:2*sizeof(int) atIndex:3];

                status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
                BREAK_IF(status != TNN_OK);
            } while (0);

            [encoder endEncoding];
            [context_impl commit];
            TNN_PRINT_ENCODER(context_, encoder, this);
            BREAK_IF(status != TNN_OK);
            
            if (layer_param->axis == 2) {
                offset[1] += input_height;
            } else {
                offset[0] += input_width;
            }
             
        }
    }

    return status;
}

REGISTER_METAL_ACC(Concat, LAYER_CONCAT);

} // namespace TNN_NS
