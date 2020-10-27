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

DECLARE_METAL_ACC(Pad, LAYER_PAD);

Status MetalPadLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalPadLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto layer_param     = dynamic_cast<PadLayerParam *>(param_);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalPadParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);

        metal_params.pad_l = layer_param->pads[0];
        metal_params.pad_r = layer_param->pads[1];
        metal_params.pad_t = layer_param->pads[2];
        metal_params.pad_b = layer_param->pads[3];
        metal_params.pad_c_b = layer_param->pads[4];
        metal_params.pad_c_e = layer_param->pads[5];
        metal_params.value = layer_param->value;
        metal_params.input_channel = dims_input[1];

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalPadParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalPadLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                           const std::vector<Blob *> &outputs,
                                           MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, false);
    return TNN_OK;
}

std::string MetalPadLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PadLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return "";
    }
    int pad_type = layer_param->type;
    bool pad_const_specilized = ((layer_param->pads[4])%4 == 0) && (inputs[0]->GetBlobDesc().dims[1]%4 == 0);

    string kernel_name = "";
    if (pad_type == 1) {
        kernel_name = "pad_reflect_common";
    } else if (pad_type == 0 && pad_const_specilized) {
        kernel_name = "pad_const_channel4";
    } else if (pad_type == 0){
        kernel_name = "pad_const_common";
    } else {
        LOGE("Error: layer param is not supported: type:%d\n", pad_type);
    }
    return kernel_name;
}

Status MetalPadLayerAcc::SetKernelEncoderParam(
    id<MTLComputeCommandEncoder> encoder,
    const std::vector<Blob *> &inputs,
    const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

Status MetalPadLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs) {
    auto data_type = outputs[0]->GetBlobDesc().data_type;
    auto data_type_str = DataTypeUtils::GetDataTypeString(data_type);
    if (data_type != DATA_TYPE_FLOAT && data_type != DATA_TYPE_HALF) {
        LOGE("MetalLayerAcc: DataType must be float or half\n");
        return Status(TNNERR_LAYER_ERR, "MetalLayerAcc: DataType must be float or half");
    }
    
    auto layer_param     = dynamic_cast<PadLayerParam *>(param_);
    int pad_type     = layer_param->type;
    bool pad_const_specilized = ((layer_param->pads[4])%4 == 0) && (inputs[0]->GetBlobDesc().dims[1]%4 == 0);

    MTLSize threads;
    auto status = ComputeThreadSize(inputs, outputs, threads);
    if (status != TNN_OK) {
        return status;
    }
    
    string kernel_name = "invalid";
    if (pad_type == 1) {
        kernel_name = "pad_reflect_common";
    } else if (pad_type == 0 && pad_const_specilized) {
        kernel_name = "pad_const_channel4";
    } else if (pad_type == 0){
        kernel_name = "pad_const_common";
    } else {
        LOGE("Error: layer param is not supported: type:%d\n", pad_type);
        return Status(TNNERR_PARAM_ERR, "Error: layer param is not supported");
    }
    
    auto context_impl = context_->getMetalContextImpl();
    auto encoder = [context_impl encoder];
    encoder.label = GetKernelLabel();
    
    do {
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

REGISTER_METAL_ACC(Pad, LAYER_PAD);

} // namespace TNN_NS
