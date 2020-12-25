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

DECLARE_METAL_ACC(Permute, LAYER_PERMUTE);

const static std::map<unsigned int, std::string> kernels = {
    {0x0231, "permute_to_nhwc"},
    {0x0213, "permute_to_nhcw"},
    {0x0312, "permute_to_nwch"},
    {0x0321, "permute_to_nwhc"},
    {0x1230, "permute_to_chwn"},
    {0x0123, "permute_copy"}
};

unsigned int GetPermuteOrderKey(const std::vector<int>& orders) {
    constexpr static unsigned int keys[4] = {
        0x1000, 0x0100, 0x010, 0x0001};
    unsigned int order_id = orders[0]*keys[0] + orders[1]*keys[1] + \
                             orders[2]*keys[2] + orders[3]*keys[3];
    return order_id;
}

bool isPermuteOrderSupported(const std::vector<int>& orders) {
    auto order_id = GetPermuteOrderKey(orders);
    switch(order_id) {
        case 0x0231: return true;
        case 0x0213: return true;
        case 0x0312: return true;
        case 0x0321: return true;
        case 0x1230: return true;
        case 0x0123: return true;
    }
    return false;
}

Status MetalPermuteLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalPermuteLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PermuteLayerParam *>(param_);
    auto order_id = GetPermuteOrderKey(layer_param->orders);
    if (kernels.count(order_id))
        return  MetalLayerAcc::AllocateBufferParam(inputs, outputs);
    
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    PermuteLayerParam* params = dynamic_cast<PermuteLayerParam *>(param_);
    
    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    {
        MetalPermuteParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
    
        const static int dims_cnt = 4;
        int input_strides[dims_cnt] = {
            metal_params.input_slice * metal_params.input_size,
            metal_params.input_size,
            metal_params.input_width,
            1
        };
        
        int output_strides[dims_cnt] = {0, 0, 0, 0};
        const auto& orders = params->orders;
        for(int i=0; i<dims_cnt; ++i) {
            output_strides[i] = input_strides[orders[i]];
            if (orders[i] == 1) {
                metal_params.channel_dim = i;
            }
            if (i == 1) {
                metal_params.channel_dim_size = dims_input[orders[i]];
            }
            metal_params.strides[i] = output_strides[i];
        }
        
        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                            length:sizeof(MetalPermuteParams)
                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    
    return TNN_OK;
}

Status MetalPermuteLayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

std::string MetalPermuteLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PermuteLayerParam *>(param_);
    auto order_id = GetPermuteOrderKey(layer_param->orders);
    if (kernels.count(order_id)) {
        printf("xxxxx:%s\n", kernels.at(order_id).c_str());
        return kernels.at(order_id);
    }
    printf("xxxxx:permute_common\n");
    return "permute_common";
}

Status MetalPermuteLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, false);
    return TNN_OK;
}

Status MetalPermuteLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PermuteLayerParam *>(param_);
    if (!layer_param) {
        return Status(TNNERR_PARAM_ERR, "PermuteLayerParam is nil");
    }
    if (layer_param->orders.size() != 4) {
        return Status(TNNERR_PARAM_ERR, "PermuteLayerParam: orders size is invalid");
    }
    
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(Permute, LAYER_PERMUTE);

} // namespace TNN_NS
