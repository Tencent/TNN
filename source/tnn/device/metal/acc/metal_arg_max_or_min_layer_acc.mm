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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

// @brief arg_max_or_argmin layer metal acc
class MetalArgMaxOrMinLayerAcc : public MetalLayerAcc {
public:
    Status Init(Context *context, LayerParam *param, LayerResource *resource,
                           const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    Status AllocateBufferParam(const std::vector<Blob*>& inputs, const std::vector<Blob*>& outputs);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    std::string KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    Status ComputeThreadSize(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs, MTLSize &size);
};

Status MetalArgMaxOrMinLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                           const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = AbstractLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);

    context_ = dynamic_cast<MetalContext *>(context);

    param_    = param;
    resource_ = resource;

#if TNN_METAL_FULL_PRECISION
    inputs[0]->GetBlobDesc().data_type  = DATA_TYPE_FLOAT;
#else
    inputs[0]->GetBlobDesc().data_type  = DATA_TYPE_HALF;
#endif
    outputs[0]->GetBlobDesc().data_type = DATA_TYPE_INT32;

    status = ReloadConstantBlobs(inputs, false);
    RETURN_ON_NEQ(status, TNN_OK);

    return Reshape(inputs, outputs);
}

std::string MetalArgMaxOrMinLayerAcc::KernelName(const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ArgMaxOrMinLayerParam *>(param_);
    if (param->axis == 1) {
        // channel
        return "argmax_or_min_channel";
    } else {
        return "argmax_or_min_common";
    }
    return "";
}

Status MetalArgMaxOrMinLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                        const std::vector<Blob *> &outputs) {
    Status status = TNN_OK;
    id<MTLDevice> device          = [TNNMetalDeviceImpl sharedDevice];
    auto param = dynamic_cast<ArgMaxOrMinLayerParam *>(param_);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto input_channel = dims_input[1];
    dims_input[1] = UP_DIV(input_channel, 4);
    // buffer_param_
    {
        MetalArgMaxOrMinParams metal_params;
        metal_params.input_channel = input_channel;
        metal_params.mode = param->mode;
        auto axis = param->axis;
        metal_params.reduce_size = dims_input[axis];
        metal_params.outer_size  = DimsVectorUtils::Count(dims_input, 0, axis);
        metal_params.inner_size  = DimsVectorUtils::Count(dims_input, axis+1);

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalArgMaxOrMinParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return status;
}

Status MetalArgMaxOrMinLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    Status status = TNN_OK;
    auto param = dynamic_cast<ArgMaxOrMinLayerParam *>(param_);
    auto axis = param->axis;

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto slice = UP_DIV(dims_input[1], 4);
    dims_input[1] = slice;

    auto outer_size  = DimsVectorUtils::Count(dims_input, 0, axis);
    auto inner_size  = DimsVectorUtils::Count(dims_input, axis+1);
    size = MTLSizeMake(inner_size, outer_size, 1);
    return status;
}

Status MetalArgMaxOrMinLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto data_type = outputs[0]->GetBlobDesc().data_type;
    if (data_type != DATA_TYPE_INT32) {
        LOGE("MetalArgMaxOrMinLayerAcc: output DataType must be int32\n");
        return Status(TNNERR_LAYER_ERR, "MetalArgMaxOrMinLayerAcc: output DataType must be int32");
    }

    auto param = dynamic_cast<ArgMaxOrMinLayerParam *>(param_);
    if (param->axis <= 1 && param->keep_dims == 0) {
        LOGE("MetalArgMaxOrMinLayerAcc: axis<1 and keep_dims=0 not supported!\n");
        return Status(TNNERR_LAYER_ERR, "MetalArgMaxOrMinLayerAcc: axis<1 and keep_dims=0 not supported!");
    }

    auto context_impl = context_->getMetalContextImpl();
    auto encoder = [context_impl encoder];
    
    MTLSize threads;
    auto status = ComputeThreadSize(inputs, outputs, threads);
    RETURN_ON_NEQ(status, TNN_OK);
    
    do {
        auto kernel_name = KernelName(inputs, outputs);
        MetalBandwidth bandwidth;
        status = [context_impl load:[NSString stringWithUTF8String:kernel_name.c_str()]
                            encoder:encoder
                          bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
        
        status = SetKernelEncoderParam(encoder, inputs, outputs);
        BREAK_IF(status != TNN_OK);
        status = [context_impl dispatchEncoder:encoder threads:threads bandwidth:bandwidth];
        BREAK_IF(status != TNN_OK);
    } while (0);

    [encoder endEncoding];
    
    if (status == TNN_OK) {
        [context_impl commit];
        TNN_PRINT_ENCODER(context_, encoder, this);
    }
    return status;
}

REGISTER_METAL_ACC(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);
REGISTER_METAL_LAYOUT(LAYER_ARG_MAX_OR_MIN, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
