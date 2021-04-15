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
    bool specialized_ = false;
};

Status MetalConcatLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Init(context, param, resource, inputs, outputs);
}

MetalConcatLayerAcc::~MetalConcatLayerAcc() {}

Status MetalConcatLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];

    auto layer_param = dynamic_cast<ConcatLayerParam *>(param_);
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    auto axis = layer_param->axis;
    specialized_ = axis == 1 && inputs.size() == 2;

    if (specialized_) {
        // specialized kernel
        auto dims_input_0 = inputs[0]->GetBlobDesc().dims;
        auto dims_input_1 = inputs[1]->GetBlobDesc().dims;
        MetalConcatParams metal_params;

        metal_params.input_size      = DimsFunctionUtils::GetDim(dims_input_0, 2) * DimsFunctionUtils::GetDimProduct(dims_input_0, 3);
        metal_params.input_channel_0 = dims_input_0[1];
        metal_params.input_slice_0   = UP_DIV(dims_input_0[1], 4);

        metal_params.input_channel_1 = dims_input_1[1];
        metal_params.input_slice_1   = UP_DIV(dims_input_1[1], 4);

        metal_params.output_channel = dims_output[1];
        metal_params.output_size    = DimsFunctionUtils::GetDim(dims_output, 2) * DimsFunctionUtils::GetDimProduct(dims_output, 3);
        metal_params.output_slice   = UP_DIV(dims_output[1], 4);

        metal_params.batch = dims_output[0];

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalConcatParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    } else {
        dims_output[1] = UP_DIV(dims_output[1], 4);
        MetalConcatParamV2 metal_params;
        metal_params.outer_size = DimsVectorUtils::Count(dims_output, 0, axis);
        metal_params.inner_size = DimsFunctionUtils::GetDimProduct(dims_output, axis+1);
        metal_params.axis_size  = DimsFunctionUtils::GetDim(dims_output, axis);

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalConcatParamV2)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }

    return TNN_OK;
}

Status MetalConcatLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<ConcatLayerParam *>(param_);

    auto output = outputs[0];

    DataType data_type       = output->GetBlobDesc().data_type;
    string data_type_str     = DataTypeUtils::GetDataTypeString(data_type);
    const int data_type_size = DataTypeUtils::GetBytesSize(data_type);
    
    int offset = 0;
    bool on_channel = layer_param->axis == 1;
    Status status = TNN_OK;

    MetalBandwidth bandwidth;
    auto context_impl = context_->getMetalContextImpl();
    auto encoder = [context_impl encoder];
    encoder.label = GetKernelLabel();

    if (specialized_) {
        do {
            auto input_0 = inputs[0];
            auto input_1 = inputs[1];
            status = [context_impl load:@"concat_axis_1_common"
                                encoder:encoder
                              bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
            
            auto dims_output   = output->GetBlobDesc().dims;
            auto output_width  = DimsFunctionUtils::GetDimProduct(dims_output, 3);
            auto output_height = DimsFunctionUtils::GetDim(dims_output, 2);
            auto output_slice  = UP_DIV(dims_output[1], 4);
            auto batch         = dims_output[0];
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
        return status;
    }

    for (int i = 0; i < inputs.size(); i++) {
        auto input_dims = inputs[i]->GetBlobDesc().dims;
        int axis_size = DimsFunctionUtils::GetDim(input_dims, layer_param->axis);
        if (on_channel) {
            axis_size = UP_DIV(axis_size, 4);
        }
        int channel_size = DimsFunctionUtils::GetDim(input_dims, 1);
        do {
            if (layer_param->axis == 1) {
                status = [context_impl load:@"concat_axis_1"
                                    encoder:encoder
                                    bandwidth:bandwidth];
            } else {
                status = [context_impl load:@"concat_common"
                                    encoder:encoder
                                    bandwidth:bandwidth];
            }
            BREAK_IF(status != TNN_OK);

            MTLSize threads;
            GetSingleAxisSplitSize(input_dims, layer_param->axis, threads, false);

            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)inputs[i]->GetHandle().base
                        offset:(NSUInteger)inputs[i]->GetHandle().bytes_offset
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                        offset:(NSUInteger)output->GetHandle().bytes_offset
                       atIndex:1];
            [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
            [encoder setBytes:&offset       length:sizeof(int) atIndex:3];
            [encoder setBytes:&axis_size    length:sizeof(int) atIndex:4];
            [encoder setBytes:&channel_size length:sizeof(int) atIndex:5];

            status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
            BREAK_IF(status != TNN_OK);
        } while (0);
        offset += DimsFunctionUtils::GetDim(input_dims, layer_param->axis);;  
    }
    [encoder endEncoding];
    [context_impl commit];
    TNN_PRINT_ENCODER(context_, encoder, this);

    return status;
}

REGISTER_METAL_ACC(Concat, LAYER_CONCAT);
REGISTER_METAL_LAYOUT(LAYER_CONCAT, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
